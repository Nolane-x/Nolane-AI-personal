from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.personal_protocol import PersonalizationSplitPolicy, build_personalization_protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--output", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--min-examples", type=int, default=6)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {output}")

    examples = load_jsonl(dataset)
    protocol = build_personalization_protocol(
        examples,
        dataset_sha256=sha256_file(dataset),
        policy=PersonalizationSplitPolicy(
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
            min_examples=args.min_examples,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(protocol["counts"], sort_keys=True))
    print(f"protocol_sha256={protocol['protocol_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
