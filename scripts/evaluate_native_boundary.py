from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.anchor_artifact import build_anchor_model
from nolane_personal.heldout_group_robustness import HeldoutGroupRobustnessPolicy, assess_group_robustness
from nolane_personal.latent import LatentStore
from nolane_personal.native_artifact import build_native_boundary_model
from nolane_personal.native_court import run_with_decoder_call_count
from nolane_personal.native_evaluation import NativeQualityEvidence, NativeQualityThresholds, decide_native_quality
from nolane_personal.native_training import mean_encoded_nll as native_mean_nll
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.state_space_training import mean_encoded_nll as anchor_mean_nll
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--native-boundary", default="runtime-data/l15-native-boundary/native-nolane-boundary.pt")
    parser.add_argument("--l14-anchor", default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt")
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
    native, native_meta = build_native_boundary_model(
        qwen.model,
        args.native_boundary,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    l14, l14_meta = build_anchor_model(
        qwen.model,
        args.l14_anchor,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )

    test = encode(qwen.tokenizer, test_examples, latent.values, args.max_length)
    anchor = encode(qwen.tokenizer, anchor_examples, latent.values, args.max_length)
    before = parameter_guard_snapshot(qwen.model)

    baseline_values = [
        native_mean_nll(native, [row], native_enabled=False)
        for row in test
    ]
    l14_values = [
        anchor_mean_nll(l14, [row], cortex_enabled=True)
        for row in test
    ]
    native_values = [
        native_mean_nll(native, [row], native_enabled=True)
        for row in test
    ]
    baseline = sum(baseline_values) / max(1, len(baseline_values))
    l14_nll = sum(l14_values) / max(1, len(l14_values))
    native_nll = sum(native_values) / max(1, len(native_values))
    anchor_base = native_mean_nll(native, anchor, native_enabled=False)
    anchor_native = native_mean_nll(native, anchor, native_enabled=True)

    import torch
    ids = torch.tensor([test[0][0]], dtype=torch.long, device=qwen.device) if test else None
    if ids is not None:
        _out, forward_calls, _per_layer = run_with_decoder_call_count(
            qwen.model,
            lambda: native.forward(input_ids=ids, state=None),
        )
        generated_a, generation_calls, _ = run_with_decoder_call_count(
            qwen.model,
            lambda: native.generate(
                input_ids=ids,
                max_new_tokens=2,
                do_sample=False,
                eos_token_id=None,
            ),
        )
        generated_b = native.generate(
            input_ids=ids,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
        )
        generation_ok = bool(torch.equal(generated_a, generated_b))
        scan_ok = native.prompt_scan_equivalent(ids)
    else:
        forward_calls = generation_calls = 1
        generation_ok = scan_ok = False

    after = parameter_guard_snapshot(qwen.model)
    evidence = NativeQualityEvidence(
        test_examples=len(test),
        anchor_examples=len(anchor),
        qwen_decoder_layers_total=int(qwen.model.config.num_hidden_layers),
        qwen_decoder_calls_forward=int(forward_calls),
        qwen_decoder_calls_generation=int(generation_calls),
        baseline_nll=baseline,
        l14_nll=l14_nll,
        native_nll=native_nll,
        improvement_vs_base=baseline-native_nll,
        degradation_vs_l14=native_nll-l14_nll,
        anchor_baseline_nll=anchor_base,
        anchor_native_nll=anchor_native,
        anchor_nll_regression=anchor_native-anchor_base,
        prompt_scan_equivalence_passed=scan_ok,
        native_generation_passed=generation_ok,
        cortex_parameters=native.trainable_parameter_count(),
        qwen_model_unchanged=before == after,
        qwen_gradients_seen=sum(1 for p in qwen.model.parameters() if p.grad is not None),
    )
    decision = decide_native_quality(evidence)
    group_robustness = assess_group_robustness(
        protocol,
        split="test",
        reference_values=l14_values,
        candidate_values=native_values,
        policy=HeldoutGroupRobustnessPolicy(
            max_worst_group_regression=NativeQualityThresholds().max_degradation_vs_l14,
        ),
    )
    result = {
        "schema": "NOLANE-L15-NATIVE-BOUNDARY-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "native_checkpoint_sha256": native_meta["checkpoint_sha256"],
        "l14_checkpoint_sha256": l14_meta["checkpoint_sha256"],
        "spec_sha256": native_meta["spec_sha256"],
        "decision": asdict(decision),
        "group_robustness": group_robustness,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite native quality evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if (
        decision.status == "NATIVE_BOUNDARY_QUALITY_PASS"
        and group_robustness["status"] == "PASS"
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
