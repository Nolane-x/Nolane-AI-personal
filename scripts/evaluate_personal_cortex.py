from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_artifact import build_personal_cortex
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_evaluation import (
    PersonalQualityEvidence,
    decide_personal_quality,
    mean_encoded_nll,
)
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
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
    result = []
    for example in examples:
        ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        result.append((ids, labels, example.latent or latent, example.weight))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--adapter", default="runtime-data/l6-personal-cortex/personal-cortex-adapter.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--anchor", default=str(ROOT / "research/personalization-general-anchor.jsonl"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_sha = sha256_file(dataset_path)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(protocol, dataset_sha256=dataset_sha)
    test_examples = examples_for_split(examples, protocol, "test")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    qwen = QwenCortex(args.model, device=args.device)
    personal, metadata = build_personal_cortex(
        qwen.model,
        args.adapter,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    if metadata.get("dataset_fingerprint") != protocol.get("protocol_sha256"):
        raise SystemExit("trained adapter personalization-protocol mismatch")

    test_encoded = encode_examples(qwen.tokenizer, test_examples, latent.values, args.max_length)
    anchor_examples = load_jsonl(args.anchor)
    anchor_encoded = encode_examples(qwen.tokenizer, anchor_examples, latent.values, args.max_length)

    base_before = parameter_guard_snapshot(qwen.model)
    baseline_nll = mean_encoded_nll(personal, test_encoded, personalized=False)
    personal_nll = mean_encoded_nll(personal, test_encoded, personalized=True)
    anchor_baseline = mean_encoded_nll(personal, anchor_encoded, personalized=False)
    anchor_personal = mean_encoded_nll(personal, anchor_encoded, personalized=True)
    base_after = parameter_guard_snapshot(qwen.model)

    evidence = PersonalQualityEvidence(
        test_examples=len(test_encoded),
        anchor_examples=len(anchor_encoded),
        baseline_nll=baseline_nll,
        personal_nll=personal_nll,
        nll_improvement=baseline_nll - personal_nll,
        anchor_baseline_nll=anchor_baseline,
        anchor_personal_nll=anchor_personal,
        anchor_nll_regression=anchor_personal - anchor_baseline,
        adapter_parameters=personal.trainable_parameter_count(),
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=sum(1 for p in qwen.model.parameters() if p.grad is not None),
    )
    decision = decide_personal_quality(evidence)
    result = {
        "schema": "NOLANE-L6-PERSONAL-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "protocol_sha256": protocol["protocol_sha256"],
        "adapter_checkpoint_sha256": metadata["checkpoint_sha256"],
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "PERSONAL_QUALITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
