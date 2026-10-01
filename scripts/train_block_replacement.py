from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.replacement_cortex import ReplacementCortexConfig, RecurrentReplacementCortex
from nolane_personal.replacement_training import (
    ReplacementTrainingConfig,
    save_trained_replacement,
    train_encoded_examples,
)
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_layers(raw: str | None, total: int) -> tuple[int, ...]:
    if raw:
        layers = tuple(sorted(set(int(x.strip()) for x in raw.split(",") if x.strip())))
    else:
        layers = tuple(sorted(set((max(1, total // 3), min(total - 2, (2 * total) // 3)))))
    if not layers or layers[0] < 0 or layers[-1] >= total:
        raise ValueError("replacement layer index out of range")
    if len(layers) >= total:
        raise ValueError("cannot replace all decoder layers")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output-dir", default="runtime-data/l9-block-replacement")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--layers", default=None)
    parser.add_argument("--recurrent-dim", type=int, default=32)
    parser.add_argument("--distill-epochs", type=int, default=2)
    parser.add_argument("--task-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--distill-learning-rate", type=float, default=3e-3)
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

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    total_layers = int(qwen.model.config.num_hidden_layers)
    layers = choose_layers(args.layers, total_layers)
    replacement = RecurrentBlockReplacement(
        int(qwen.model.config.hidden_size),
        BlockReplacementConfig(
            latent_dim=32,
            recurrent_dim=args.recurrent_dim,
            max_layers=max(64, total_layers),
            initial_gate=0.05,
        ),
        seed=0,
    )
    if replacement.parameter_count() > 100_000:
        raise SystemExit(f"replacement exceeds 100K parameter cap: {replacement.parameter_count()}")
    replacement.to(qwen.device)
    cortex = RecurrentReplacementCortex(
        qwen.model,
        replacement,
        latent.values,
        config=ReplacementCortexConfig(layer_indices=layers, carry_recurrent_state=False),
    )

    encoded = []
    for example in train_examples:
        ids, labels = encode_chat_example(
            qwen.tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=args.max_length,
        )
        encoded.append((ids, labels, example.latent or latent.values, example.weight))

    receipt = train_encoded_examples(
        cortex,
        encoded,
        config=ReplacementTrainingConfig(
            distill_epochs=args.distill_epochs,
            task_epochs=args.task_epochs,
            learning_rate=args.learning_rate,
            distill_learning_rate=args.distill_learning_rate,
            max_grad_norm=1.0,
        ),
    )
    manifest = save_trained_replacement(
        args.output_dir,
        cortex,
        receipt,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps({"training": receipt.to_dict(), "artifact": manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt.base_model_unchanged and receipt.base_gradients_seen == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
