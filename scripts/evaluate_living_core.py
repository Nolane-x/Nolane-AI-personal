from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.evaluation import evaluate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runtime-data/living.db")
    parser.add_argument("--protocol", default="runtime-data/replay-protocol-v1.json")
    parser.add_argument("--checkpoint", default="runtime-data/living-core-dev/living-core.pt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate_checkpoint(args.db, args.protocol, args.checkpoint)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["promotion"]["status"] == "PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
