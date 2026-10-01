from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.latent import LatentStore
from nolane_personal.native_artifact import save_native_boundary_artifact
from nolane_personal.native_boundary import NativeBoundaryConfig, NativeNolaneBoundaryModel
from nolane_personal.native_spec import verify_native_boundary_spec
from nolane_personal.native_training import NativeTrainingConfig, train_native_boundary
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
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--spec", default="runtime-data/l15-native-boundary-spec.json")
    parser.add_argument("--output-dir", default="runtime-data/l15-native-boundary")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--virtual-steps", type=int, default=8)
    parser.add_argument("--max-virtual-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    train_examples = examples_for_split(examples, protocol, "train")
    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    verify_native_boundary_spec(
        spec,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )

    qwen = QwenCortex(args.model, device=args.device)
    cortex = DeepRecurrentStateSpaceCortex(
        int(qwen.model.config.hidden_size),
        DeepRecurrentCortexConfig(
            latent_dim=32,
            state_dim=args.state_dim,
            virtual_steps=args.virtual_steps,
            max_virtual_steps=args.max_virtual_steps,
        ),
        seed=0,
    )
    if cortex.parameter_count() > 100_000:
        raise SystemExit(f"native cortex exceeds 100K cap: {cortex.parameter_count()}")
    cortex.to(qwen.device)
    model = NativeNolaneBoundaryModel(
        qwen.model,
        cortex,
        latent.values,
        config=NativeBoundaryConfig(carry_recurrent_state=False),
    )
    encoded = encode(qwen.tokenizer, train_examples, latent.values, args.max_length)
    receipt = train_native_boundary(
        model,
        encoded,
        spec,
        config=NativeTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            distill_weight=args.distill_weight,
            distill_temperature=args.distill_temperature,
        ),
    )
    manifest = save_native_boundary_artifact(
        args.output_dir,
        model,
        receipt,
        spec,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps(
        {"training": receipt.to_dict(), "artifact": manifest},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
    return 0 if receipt.qwen_model_unchanged and receipt.qwen_gradients_seen == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
