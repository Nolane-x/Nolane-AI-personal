from __future__ import annotations

import argparse
import json

from nolane_personal.training import train_from_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--output-dir", default="runtime-data/living-core-dev")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--truncation", type=int, default=16)
    parser.add_argument("--return-horizon-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    manifest = train_from_store(
        args.db,
        args.output_dir,
        protocol_path=args.protocol,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        truncation=args.truncation,
        return_horizon_seconds=args.return_horizon_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
