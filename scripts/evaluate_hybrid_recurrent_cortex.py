from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.hybrid_artifact import load_hybrid_artifact
from nolane_personal.hybrid_evaluation import evaluate_hybrid
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
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
    parser.add_argument("--hybrid", default="runtime-data/l7-hybrid-recurrent/hybrid-recurrent.pt")
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

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    qwen = QwenCortex(args.model, device=args.device)
    cortex, metadata = load_hybrid_artifact(
        args.hybrid,
        expected_base_model_fingerprint=fingerprint,
        expected_hidden_size=int(qwen.model.config.hidden_size),
        latent=latent.values,
        model=qwen.model,
    )
    cortex.mixer.to(qwen.device).eval()

    test_encoded = encode_examples(
        qwen.tokenizer,
        test_examples,
        latent.values,
        args.max_length,
    )
    anchor_examples = load_jsonl(args.anchor)
    anchor_encoded = encode_examples(
        qwen.tokenizer,
        anchor_examples,
        latent.values,
        args.max_length,
    )
    decision = evaluate_hybrid(cortex, test_encoded, anchor_encoded)
    result = {
        "schema": "NOLANE-L7-HYBRID-QUALITY-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "protocol_sha256": protocol["protocol_sha256"],
        "hybrid_checkpoint_sha256": metadata["checkpoint_sha256"],
        "mixer_digest": metadata["mixer_digest"],
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
    return 0 if decision.status == "HYBRID_QUALITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
