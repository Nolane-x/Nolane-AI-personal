from __future__ import annotations

import argparse
import json

from nolane_personal.training import train_from_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--output-dir", default="runtime-data/living-core-dev")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--truncation", type=int, default=16)
    args = parser.parse_args()
    manifest = train_from_store(
        args.db,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        truncation=args.truncation,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
