from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from nolane_personal.island_artifact import save_island_artifact
from nolane_personal.island_cortex import IslandCortexConfig, RecurrentIslandCortex
from nolane_personal.island_plan import verify_island_plan
from nolane_personal.island_replacement import TransformerIsland
from nolane_personal.island_training import IslandTrainingConfig, train_island_stages
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
    parser.add_argument("--plan", default="runtime-data/l11-island-plan.json")
    parser.add_argument("--output-dir", default="runtime-data/l11-recurrent-islands")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--recurrent-dim", type=int, default=32)
    parser.add_argument("--distill-epochs-per-stage", type=int, default=2)
    parser.add_argument("--task-epochs-per-stage", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--distill-learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-dev-regression", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(protocol, dataset_sha256=sha256_file(dataset_path))
    train_examples = examples_for_split(examples, protocol, "train")
    dev_examples = examples_for_split(examples, protocol, "dev")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    verify_island_plan(
        plan,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )

    qwen = QwenCortex(args.model, device=args.device)
    total_layers = int(qwen.model.config.num_hidden_layers)
    if int(plan["total_layers"]) != total_layers:
        raise SystemExit("island plan/model layer-count mismatch")

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
        raise SystemExit(f"island replacement exceeds 100K parameter cap: {replacement.parameter_count()}")
    replacement.to(qwen.device)
    first = tuple(
        TransformerIsland(int(row["start"]), int(row["end"]))
        for row in plan["stages"][0]["islands"]
    )
    cortex = RecurrentIslandCortex(
        qwen.model,
        replacement,
        latent.values,
        config=IslandCortexConfig(islands=first, carry_recurrent_state=False),
    )
    train = encode(qwen.tokenizer, train_examples, latent.values, args.max_length)
    dev = encode(qwen.tokenizer, dev_examples, latent.values, args.max_length)
    receipt = train_island_stages(
        cortex,
        train,
        dev,
        plan,
        config=IslandTrainingConfig(
            distill_epochs_per_stage=args.distill_epochs_per_stage,
            task_epochs_per_stage=args.task_epochs_per_stage,
            learning_rate=args.learning_rate,
            distill_learning_rate=args.distill_learning_rate,
            max_dev_regression=args.max_dev_regression,
        ),
    )
    manifest = save_island_artifact(
        args.output_dir,
        cortex,
        receipt,
        plan,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps({"training": receipt.to_dict(), "artifact": manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt.base_model_unchanged and receipt.base_gradients_seen == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
