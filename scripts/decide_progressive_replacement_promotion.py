from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.progressive_promotion import decide_progressive_promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
    resources = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    decision = decide_progressive_promotion(
        quality_status=str(quality.get("decision", {}).get("status", "")),
        quality_checkpoint_sha256=quality.get("progressive_checkpoint_sha256"),
        quality_plan_sha256=quality.get("plan_sha256"),
        resource_status=str(resources.get("decision", {}).get("status", "")),
        resource_checkpoint_sha256=resources.get("progressive_checkpoint_sha256"),
        resource_plan_sha256=resources.get("plan_sha256"),
    )
    result = {
        "schema": "NOLANE-L10-PROGRESSIVE-REPLACEMENT-PROMOTION-V1",
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite promotion decision: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "PROGRESSIVE_REPLACEMENT_PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
