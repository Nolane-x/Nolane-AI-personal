from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.bridge_cortex import LivingBridgeCortexConfig, TrainableLivingBridgeCortex
from nolane_personal.bridge_training import BridgeTrainingConfig, save_trained_bridge, train_encoded_examples
from nolane_personal.latent import LatentStore
from nolane_personal.living_bridge import CrossLayerLivingBridge, LivingBridgeConfig
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


def parse_layers(raw: str | None):
    if raw is None:
        return None
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return values or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output-dir", default="runtime-data/l7-living-bridge")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--layers", default=None)
    parser.add_argument("--token-scope", choices=["all", "last"], default="all")
    parser.add_argument("--bridge-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
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
    max_layers = max(64, int(qwen.model.config.num_hidden_layers))
    bridge = CrossLayerLivingBridge(
        hidden_size,
        LivingBridgeConfig(
            latent_dim=32,
            bridge_dim=args.bridge_dim,
            max_layers=max_layers,
            max_abs_gate=0.15,
            initial_gate=0.02,
        ),
        seed=0,
    )
    if bridge.parameter_count() > 100_000:
        raise SystemExit(f"bridge exceeds 100K parameter cap: {bridge.parameter_count()}")
    bridge.to(qwen.device)
    cortex = TrainableLivingBridgeCortex(
        qwen.model,
        bridge,
        latent.values,
        config=LivingBridgeCortexConfig(
            layer_indices=parse_layers(args.layers),
            token_scope=args.token_scope,
        ),
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
        config=BridgeTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_grad_norm=1.0,
        ),
    )
    manifest = save_trained_bridge(
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
