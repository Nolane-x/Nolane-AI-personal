from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.island_plan import (
    IslandPlanConfig,
    build_island_plan,
    calibrate_island_sensitivity,
)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output", default="runtime-data/l11-island-plan.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-fraction", type=float, default=0.35)
    parser.add_argument("--min-width", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=4)
    parser.add_argument("--max-islands", type=int, default=4)
    parser.add_argument("--edge-layers-to-keep", type=int, default=1)
    parser.add_argument("--min-gap-layers", type=int, default=1)
    parser.add_argument("--max-calibration-examples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset_sha = sha256_file(dataset_path)
    examples = load_jsonl(dataset_path)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(protocol, dataset_sha256=dataset_sha)
    train_examples = examples_for_split(examples, protocol, "train")
    calibration_examples = train_examples[: max(1, args.max_calibration_examples)]

    qwen = QwenCortex(args.model, device=args.device)
    encoded = []
    for example in calibration_examples:
        ids, _labels = encode_chat_example(
            qwen.tokenizer,
            example,
            system_prompt=SYSTEM_PROMPT,
            max_length=args.max_length,
        )
        encoded.append(ids)

    config = IslandPlanConfig(
        target_fraction=args.target_fraction,
        min_width=args.min_width,
        max_width=args.max_width,
        max_islands=args.max_islands,
        edge_layers_to_keep=args.edge_layers_to_keep,
        min_gap_layers=args.min_gap_layers,
    )
    sensitivity = calibrate_island_sensitivity(qwen.model, encoded, config=config)
    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    plan = build_island_plan(
        total_layers=int(qwen.model.config.num_hidden_layers),
        sensitivity=sensitivity,
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
        config=config,
    )

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen island plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "plan_sha256": plan["plan_sha256"],
        "target_islands": plan["target_islands"],
        "replaced_layers": plan["replaced_layers"],
        "replaced_fraction_actual": plan["replaced_fraction_actual"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
