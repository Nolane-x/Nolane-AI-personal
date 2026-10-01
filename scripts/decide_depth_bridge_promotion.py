from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.depth_bridge_promotion import decide_depth_bridge_promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
    resources = json.loads(Path(args.resources).read_text(encoding="utf-8"))

    quality_decision = quality.get("decision", {})
    resource_decision = resources.get("decision", {})
    quality_sha = quality.get("artifacts", {}).get("l8_depth_bridge")
    resource_sha = resources.get("bridge_checkpoint_sha256")

    decision = decide_depth_bridge_promotion(
        quality_status=str(quality_decision.get("status", "")),
        quality_checkpoint_sha256=quality_sha,
        resource_status=str(resource_decision.get("status", "")),
        resource_checkpoint_sha256=resource_sha,
    )
    result = {
        "schema": "NOLANE-L8-DEPTH-BRIDGE-PROMOTION-V1",
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
    return 0 if decision.status == "DEPTH_BRIDGE_PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
