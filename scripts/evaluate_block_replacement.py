from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.depth_bridge_artifact import build_bridge_cortex
from nolane_personal.depth_bridge_training import mean_encoded_nll as l8_mean_nll
from nolane_personal.hybrid_artifact import load_hybrid_artifact
from nolane_personal.hybrid_evaluation import mean_hybrid_nll
from nolane_personal.latent import LatentStore
from nolane_personal.personal_artifact import build_personal_cortex
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_evaluation import mean_encoded_nll as l6_mean_nll
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.replacement_artifact import build_replacement_cortex
from nolane_personal.replacement_evaluation import ReplacementQualityEvidence, decide_replacement_quality
from nolane_personal.replacement_training import mean_encoded_nll as l9_mean_nll
from nolane_personal.surgery import parameter_guard_snapshot
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_examples(tokenizer, examples, latent, max_length):
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
    parser.add_argument("--l9-replacement", default="runtime-data/l9-block-replacement/recurrent-block-replacement.pt")
    parser.add_argument("--l8-depth-bridge", default="runtime-data/l8-depth-bridge/depth-bridge.pt")
    parser.add_argument("--l7-hybrid", default="runtime-data/l7-hybrid-recurrent/hybrid-recurrent.pt")
    parser.add_argument("--l6-adapter", default="runtime-data/l6-personal-cortex/personal-cortex-adapter.pt")
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
    hidden_size = int(qwen.model.config.hidden_size)

    l9, l9_meta = build_replacement_cortex(
        qwen.model, args.l9_replacement, latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    l8, l8_meta = build_bridge_cortex(
        qwen.model, args.l8_depth_bridge, latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    l7, l7_meta = load_hybrid_artifact(
        args.l7_hybrid,
        expected_base_model_fingerprint=fingerprint,
        expected_hidden_size=hidden_size,
        latent=latent.values,
        model=qwen.model,
    )
    l7.mixer.to(qwen.device).eval()
    l6, l6_meta = build_personal_cortex(
        qwen.model, args.l6_adapter, latent.values,
        expected_base_model_fingerprint=fingerprint,
    )

    expected_protocol = protocol["protocol_sha256"]
    for name, meta in (
        ("L9 replacement", l9_meta),
        ("L8 depth bridge", l8_meta),
        ("L7 hybrid", l7_meta),
        ("L6 personal cortex", l6_meta),
    ):
        if meta.get("dataset_fingerprint") != expected_protocol:
            raise SystemExit(f"{name} personalization-protocol mismatch")

    test = encode_examples(qwen.tokenizer, test_examples, latent.values, args.max_length)
    anchor = encode_examples(qwen.tokenizer, anchor_examples, latent.values, args.max_length)

    before = parameter_guard_snapshot(qwen.model)
    baseline = l9_mean_nll(l9, test, replacement_enabled=False)
    l6_nll = l6_mean_nll(l6, test, personalized=True)
    l7_nll = mean_hybrid_nll(l7, test, hybrid=True)
    l8_nll = l8_mean_nll(l8, test, personalized=True)
    l9_nll = l9_mean_nll(l9, test, replacement_enabled=True)
    anchor_base = l9_mean_nll(l9, anchor, replacement_enabled=False)
    anchor_l9 = l9_mean_nll(l9, anchor, replacement_enabled=True)
    after = parameter_guard_snapshot(qwen.model)

    best_prior = min(l6_nll, l7_nll, l8_nll)
    evidence = ReplacementQualityEvidence(
        test_examples=len(test),
        anchor_examples=len(anchor),
        skipped_layers=len(l9.config.layer_indices),
        baseline_nll=baseline,
        l6_nll=l6_nll,
        l7_nll=l7_nll,
        l8_nll=l8_nll,
        replacement_nll=l9_nll,
        improvement_vs_base=baseline-l9_nll,
        best_prior_nll=best_prior,
        degradation_vs_best_prior=l9_nll-best_prior,
        anchor_baseline_nll=anchor_base,
        anchor_replacement_nll=anchor_l9,
        anchor_nll_regression=anchor_l9-anchor_base,
        replacement_parameters=l9.trainable_parameter_count(),
        base_model_unchanged=before == after,
        base_gradients_seen=sum(1 for p in qwen.model.parameters() if p.grad is not None),
    )
    decision = decide_replacement_quality(evidence)
    result = {
        "schema": "NOLANE-L9-BLOCK-REPLACEMENT-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "protocol_sha256": expected_protocol,
        "artifacts": {
            "l9_replacement": l9_meta["checkpoint_sha256"],
            "l8_depth_bridge": l8_meta["checkpoint_sha256"],
            "l7_hybrid": l7_meta["checkpoint_sha256"],
            "l6_personal_cortex": l6_meta["checkpoint_sha256"],
        },
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered+"\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "BLOCK_REPLACEMENT_QUALITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
