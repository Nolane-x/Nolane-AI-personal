from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.hybrid_artifact import save_hybrid_artifact
from nolane_personal.hybrid_training import HybridTrainingConfig, train_hybrid_encoded_examples
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.recurrent_cortex import (
    HybridCortexConfig,
    HybridRecurrentCortex,
    RecurrentCortexConfig,
    RecurrentCortexMixer,
    analytical_recurrent_parameter_count,
)
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(raw: str | None):
    if raw is None:
        return None
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return values or None


def encode(tokenizer, examples, default_latent, max_length):
    rows = []
    for example in examples:
        ids, labels = encode_chat_example(
            tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=max_length,
        )
        rows.append((ids, labels, example.latent or default_latent, example.weight))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output", default="runtime-data/l7-hybrid-recurrent/hybrid-recurrent.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--layers", default=None)
    parser.add_argument("--recurrent-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(protocol, dataset_sha256=sha256_file(dataset_path))
    train_examples = examples_for_split(examples, protocol, "train")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    qwen = QwenCortex(args.model, device=args.device)
    hidden_size = int(qwen.model.config.hidden_size)
    mixer_config = RecurrentCortexConfig(
        latent_dim=latent.latent_dim,
        recurrent_dim=args.recurrent_dim,
        max_abs_gate=0.10,
    )
    parameter_count = analytical_recurrent_parameter_count(hidden_size, mixer_config)
    if parameter_count > 100_000:
        raise SystemExit(f"recurrent mixer exceeds 100K parameter cap: {parameter_count}")

    mixer = RecurrentCortexMixer(hidden_size, mixer_config, seed=0).to(qwen.device)
    cortex = HybridRecurrentCortex(
        qwen.model,
        mixer,
        latent.values,
        config=HybridCortexConfig(
            layer_indices=parse_layers(args.layers),
            carry_across_calls=True,
        ),
    )
    rows = encode(qwen.tokenizer, train_examples, latent.values, args.max_length)
    receipt = train_hybrid_encoded_examples(
        cortex,
        rows,
        config=HybridTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        ),
    )
    manifest = save_hybrid_artifact(
        args.output,
        cortex,
        receipt,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps({"training": receipt.to_dict(), "artifact": manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt.base_model_unchanged and receipt.base_gradients_seen == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
