from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.progressive_artifact import build_progressive_cortex
from nolane_personal.progressive_evaluation import ProgressiveQualityEvidence, decide_progressive_quality
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.replacement_artifact import build_replacement_cortex
from nolane_personal.replacement_training import mean_encoded_nll
from nolane_personal.surgery import parameter_guard_snapshot
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


def cached_generation_check(cortex, ids, *, max_new_tokens: int = 2) -> bool:
    import torch
    device = next(cortex.model.parameters()).device
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    cortex.reset_state()
    cached = cortex.generate_cached(
        input_ids=tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=0,
    )
    cortex.reset_state()
    replay = cortex.generate_replay_safe(
        input_ids=tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=0,
    )
    return bool(torch.equal(cached, replay))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--l10-progressive", default="runtime-data/l10-progressive-replacement/progressive-block-replacement.pt")
    parser.add_argument("--l9-replacement", default="runtime-data/l9-block-replacement/recurrent-block-replacement.pt")
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
    verify_personalization_protocol(protocol, dataset_sha256=sha256_file(dataset_path))
    test_examples = examples_for_split(examples, protocol, "test")
    anchor_examples = load_jsonl(args.anchor)

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    l10, l10_meta = build_progressive_cortex(
        qwen.model,
        args.l10_progressive,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    l9, l9_meta = build_replacement_cortex(
        qwen.model,
        args.l9_replacement,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    if l9_meta.get("dataset_fingerprint") != protocol["protocol_sha256"]:
        raise SystemExit("L9 personalization-protocol mismatch")

    test = encode(qwen.tokenizer, test_examples, latent.values, args.max_length)
    anchor = encode(qwen.tokenizer, anchor_examples, latent.values, args.max_length)
    before = parameter_guard_snapshot(qwen.model)
    baseline = mean_encoded_nll(l10, test, replacement_enabled=False)
    l9_nll = mean_encoded_nll(l9, test, replacement_enabled=True)
    l10_nll = mean_encoded_nll(l10, test, replacement_enabled=True)
    anchor_base = mean_encoded_nll(l10, anchor, replacement_enabled=False)
    anchor_l10 = mean_encoded_nll(l10, anchor, replacement_enabled=True)
    cached_ok = cached_generation_check(l10, test[0][0]) if test else False
    after = parameter_guard_snapshot(qwen.model)

    receipt = l10_meta.get("training_receipt") or {}
    evidence = ProgressiveQualityEvidence(
        test_examples=len(test),
        anchor_examples=len(anchor),
        total_layers=int(qwen.model.config.num_hidden_layers),
        replaced_layers=len(l10.config.layer_indices),
        stages_completed=int(receipt.get("stages_accepted", 0)),
        baseline_nll=baseline,
        l9_nll=l9_nll,
        progressive_nll=l10_nll,
        improvement_vs_base=baseline-l10_nll,
        degradation_vs_l9=l10_nll-l9_nll,
        anchor_baseline_nll=anchor_base,
        anchor_progressive_nll=anchor_l10,
        anchor_nll_regression=anchor_l10-anchor_base,
        replacement_parameters=l10.trainable_parameter_count(),
        cached_generation_passed=cached_ok,
        base_model_unchanged=before == after,
        base_gradients_seen=sum(1 for p in qwen.model.parameters() if p.grad is not None),
    )
    decision = decide_progressive_quality(evidence)
    result = {
        "schema": "NOLANE-L10-PROGRESSIVE-REPLACEMENT-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "progressive_checkpoint_sha256": l10_meta["checkpoint_sha256"],
        "l9_checkpoint_sha256": l9_meta["checkpoint_sha256"],
        "plan_sha256": l10_meta["plan_sha256"],
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite quality evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "PROGRESSIVE_REPLACEMENT_QUALITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
