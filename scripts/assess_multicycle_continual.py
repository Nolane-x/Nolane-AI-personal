from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.multicycle_continual import (
    MultiCycleContinualPolicy,
    assess_multicycle_continual_chain,
)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cycle",
        action="append",
        required=True,
        help="Path to an L31 run receipt. Repeat for each cycle in order.",
    )
    parser.add_argument("--long-horizon", required=True)
    parser.add_argument("--min-cycles", type=int, default=2)
    parser.add_argument(
        "--allow-reused-adaptation-protocol",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite multicycle receipt: {output}"
        )

    cycles = [load_json(path) for path in args.cycle]
    long_horizon = load_json(args.long_horizon)
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=long_horizon,
        policy=MultiCycleContinualPolicy(
            min_cycles=args.min_cycles,
            require_unique_adaptation_protocols=(
                not args.allow_reused_adaptation_protocol
            ),
        ),
    )
    rendered = json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
