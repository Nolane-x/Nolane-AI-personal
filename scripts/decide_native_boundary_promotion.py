from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.native_promotion import decide_native_promotion
from nolane_personal.heldout_group_robustness import effective_quality_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
    resources = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    group = quality.get("group_robustness")
    effective_status = effective_quality_status(quality)
    decision = decide_native_promotion(
        quality_status=effective_status,
        quality_checkpoint_sha256=quality.get("native_checkpoint_sha256"),
        quality_spec_sha256=quality.get("spec_sha256"),
        resource_status=str(resources.get("decision", {}).get("status", "")),
        resource_checkpoint_sha256=resources.get("native_checkpoint_sha256"),
        resource_spec_sha256=resources.get("spec_sha256"),
    )
    result = {
        "schema": "NOLANE-L15-NATIVE-BOUNDARY-PROMOTION-V1",
        "group_robustness_status": group.get("status") if isinstance(group, dict) else None,
        "group_robustness_court_sha256": group.get("court_sha256") if isinstance(group, dict) else None,
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
    return 0 if decision.status == "NATIVE_BOUNDARY_PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
