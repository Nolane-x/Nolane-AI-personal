from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.state_space_artifact import save_state_space_artifact
from nolane_personal.state_space_core import SelectiveStateSpaceCortex, StateSpaceCortexConfig
from nolane_personal.state_space_model import StateSpaceModelConfig, StateSpacePersonalModel
from nolane_personal.state_space_plan import verify_state_space_plan
from nolane_personal.state_space_region import CortexRegion
from nolane_personal.state_space_training import StateSpaceTrainingConfig, train_state_space_stages
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
    parser.add_argument("--plan", default="runtime-data/l12-state-space-plan.json")
    parser.add_argument("--output-dir", default="runtime-data/l12-state-space-cortex")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--state-dim", type=int, default=32)
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
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    train_examples = examples_for_split(examples, protocol, "train")
    dev_examples = examples_for_split(examples, protocol, "dev")

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    verify_state_space_plan(
        plan,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )

    qwen = QwenCortex(args.model, device=args.device)
    total_layers = int(qwen.model.config.num_hidden_layers)
    if int(plan["total_layers"]) != total_layers:
        raise SystemExit("state-space plan/model layer-count mismatch")

    cortex = SelectiveStateSpaceCortex(
        int(qwen.model.config.hidden_size),
        StateSpaceCortexConfig(
            latent_dim=32,
            state_dim=args.state_dim,
            max_abs_gate=0.20,
            initial_gate=0.04,
        ),
        seed=0,
    )
    if cortex.parameter_count() > 100_000:
        raise SystemExit(f"state-space cortex exceeds 100K parameter cap: {cortex.parameter_count()}")
    cortex.to(qwen.device)
    first = plan["stages"][0]["region"]
    model = StateSpacePersonalModel(
        qwen.model,
        cortex,
        latent.values,
        config=StateSpaceModelConfig(
            region=CortexRegion(int(first["start"]), int(first["end"])),
            carry_recurrent_state=False,
        ),
    )

    train = encode(qwen.tokenizer, train_examples, latent.values, args.max_length)
    dev = encode(qwen.tokenizer, dev_examples, latent.values, args.max_length)
    receipt = train_state_space_stages(
        model,
        train,
        dev,
        plan,
        config=StateSpaceTrainingConfig(
            distill_epochs_per_stage=args.distill_epochs_per_stage,
            task_epochs_per_stage=args.task_epochs_per_stage,
            learning_rate=args.learning_rate,
            distill_learning_rate=args.distill_learning_rate,
            max_dev_regression=args.max_dev_regression,
        ),
    )
    manifest = save_state_space_artifact(
        args.output_dir,
        model,
        receipt,
        plan,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )
    print(json.dumps({
        "training": receipt.to_dict(),
        "artifact": manifest,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt.base_model_unchanged and receipt.base_gradients_seen == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
