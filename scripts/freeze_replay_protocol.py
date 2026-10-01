from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.replay_protocol import ReplaySplitPolicy, build_replay_protocol
from nolane_personal.store import LivingStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--output", default="runtime-data/replay-protocol-v1.json")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--min-records", type=int, default=6)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {output}")

    store = LivingStore(args.db)
    try:
        protocol = build_replay_protocol(
            store,
            ReplaySplitPolicy(
                train_fraction=args.train_fraction,
                dev_fraction=args.dev_fraction,
                min_records=args.min_records,
            ),
        )
    finally:
        store.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(protocol["counts"], sort_keys=True))
    print(f"protocol_sha256={protocol['protocol_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
