from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_cortex import PersonalCortexConfig, TrainablePersonalCortex
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.personal_training import PersonalTrainingConfig, save_trained_adapter, train_text_examples
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.surgery_candidate import load_candidate, load_model_lock, model_lock_fingerprint


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
    result = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return result or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--candidate", default="runtime-data/l5-adapter-candidate/latent-adapter.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output-dir", default="runtime-data/l6-personal-cortex")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--layers", default=None)
    parser.add_argument("--token-scope", choices=["all", "last"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--initial-effective-gate", type=float, default=0.02)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_sha = sha256_file(dataset_path)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(protocol, dataset_sha256=dataset_sha)
    train_examples = examples_for_split(examples, protocol, "train")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    cortex_model = QwenCortex(args.model, device=args.device)
    hidden_size = int(cortex_model.model.config.hidden_size)
    adapter, candidate_meta = load_candidate(
        args.candidate,
        expected_model_lock_fingerprint=fingerprint,
        expected_hidden_size=hidden_size,
    )
    adapter.to(cortex_model.device)

    config = PersonalCortexConfig(
        layer_indices=parse_layers(args.layers),
        token_scope=args.token_scope,
    )
    personal = TrainablePersonalCortex(
        cortex_model.model,
        adapter,
        latent.values,
        config=config,
    )
    receipt = train_text_examples(
        personal,
        cortex_model.tokenizer,
        train_examples,
        system_prompt=SYSTEM_PROMPT,
        default_latent=latent.values,
        config=PersonalTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            initial_effective_gate=args.initial_effective_gate,
        ),
    )
    manifest = save_trained_adapter(
        args.output_dir,
        personal,
        receipt,
        base_model_fingerprint=fingerprint,
        source_candidate_checkpoint_sha256=candidate_meta["checkpoint_sha256"],
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps({"training": receipt.to_dict(), "artifact": manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    if not receipt.base_model_unchanged or receipt.base_gradients_seen:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
