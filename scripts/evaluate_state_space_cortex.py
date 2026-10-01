from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.island_artifact import build_island_cortex
from nolane_personal.island_training import mean_encoded_nll as island_mean_nll
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.state_space_artifact import build_state_space_model
from nolane_personal.state_space_evaluation import StateSpaceQualityEvidence, decide_state_space_quality
from nolane_personal.state_space_training import mean_encoded_nll as state_space_mean_nll
from nolane_personal.surgery import parameter_guard_snapshot, resolve_transformer_layers
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(tokenizer, examples, latent, max_length):
    rows = []
    for example in examples:
        ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        rows.append((ids, labels, example.latent or latent, example.weight))
    return rows


def cached_generation_check(model, ids, *, pad_token_id: int, max_new_tokens: int = 2) -> bool:
    import torch
    device = next(model.model.parameters()).device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    model.reset_state()
    cached = model.generate_cached(
        input_ids=tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    model.reset_state()
    replay = model.generate_replay_safe(
        input_ids=tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    return bool(torch.equal(cached, replay))


def capture_region_input(model, input_ids, region_start: int):
    import torch
    layers = resolve_transformer_layers(model)
    captured = {}

    def pre_hook(_module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("decoder hidden_states missing")
        captured["hidden"] = hidden.detach()

    handle = layers[int(region_start)].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("failed to capture state-space region input")
    return captured["hidden"]


def scan_equivalence_check(model, ids, latent) -> bool:
    import torch
    device = next(model.model.parameters()).device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = capture_region_input(model.model, tensor, model.config.region.start)
    model.cortex.eval()
    with torch.no_grad():
        full, full_state, _ = model.cortex.scan(hidden, latent, state=None)
        pieces = []
        state = None
        for index in range(hidden.shape[1]):
            piece, state, _ = model.cortex.scan(
                hidden[:, index:index+1, :],
                latent,
                state=state,
            )
            pieces.append(piece)
        incremental = torch.cat(pieces, dim=1)
    return bool(
        torch.allclose(full, incremental, atol=1e-5, rtol=1e-5)
        and torch.allclose(full_state, state, atol=1e-5, rtol=1e-5)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--state-space", default="runtime-data/l12-state-space-cortex/selective-state-space-cortex.pt")
    parser.add_argument("--l11-islands", default="runtime-data/l11-recurrent-islands/recurrent-transformer-islands.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--anchor", default=str(ROOT / "research/personalization-general-anchor.jsonl"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    test_examples = examples_for_split(examples, protocol, "test")
    anchor_examples = load_jsonl(args.anchor)

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    state_space, state_meta = build_state_space_model(
        qwen.model,
        args.state_space,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    l11, l11_meta = build_island_cortex(
        qwen.model,
        args.l11_islands,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )

    test = encode(qwen.tokenizer, test_examples, latent.values, args.max_length)
    anchor = encode(qwen.tokenizer, anchor_examples, latent.values, args.max_length)
    before = parameter_guard_snapshot(qwen.model)
    baseline = state_space_mean_nll(state_space, test, cortex_enabled=False)
    l11_nll = island_mean_nll(l11, test, islands_enabled=True)
    ss_nll = state_space_mean_nll(state_space, test, cortex_enabled=True)
    anchor_base = state_space_mean_nll(state_space, anchor, cortex_enabled=False)
    anchor_ss = state_space_mean_nll(state_space, anchor, cortex_enabled=True)
    pad = qwen.tokenizer.eos_token_id if qwen.tokenizer.eos_token_id is not None else 0
    cached_ok = cached_generation_check(state_space, test[0][0], pad_token_id=pad) if test else False
    scan_ok = scan_equivalence_check(
        state_space,
        test[0][0],
        test[0][2] if test and test[0][2] is not None else latent.values,
    ) if test else False
    after = parameter_guard_snapshot(qwen.model)

    training = state_meta.get("training_receipt") or {}
    evidence = StateSpaceQualityEvidence(
        test_examples=len(test),
        anchor_examples=len(anchor),
        total_layers=int(qwen.model.config.num_hidden_layers),
        replaced_layers=state_space.replaced_layer_count(),
        stages_completed=int(training.get("stages_accepted", 0)),
        baseline_nll=baseline,
        l11_nll=l11_nll,
        state_space_nll=ss_nll,
        improvement_vs_base=baseline-ss_nll,
        degradation_vs_l11=ss_nll-l11_nll,
        anchor_baseline_nll=anchor_base,
        anchor_state_space_nll=anchor_ss,
        anchor_nll_regression=anchor_ss-anchor_base,
        cached_generation_passed=cached_ok,
        scan_equivalence_passed=scan_ok,
        cortex_parameters=state_space.trainable_parameter_count(),
        base_model_unchanged=before == after,
        base_gradients_seen=sum(1 for p in qwen.model.parameters() if p.grad is not None),
    )
    decision = decide_state_space_quality(evidence)
    result = {
        "schema": "NOLANE-L12-STATE-SPACE-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "state_space_checkpoint_sha256": state_meta["checkpoint_sha256"],
        "l11_checkpoint_sha256": l11_meta["checkpoint_sha256"],
        "plan_sha256": state_meta["plan_sha256"],
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite state-space quality evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "STATE_SPACE_CORTEX_QUALITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
