from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.standalone_promotion import decide_standalone_promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parity = json.loads(Path(args.parity).read_text(encoding="utf-8"))
    resources = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    decision = decide_standalone_promotion(
        parity_status=str(parity.get("decision", {}).get("status", "")),
        parity_checkpoint_sha256=parity.get("standalone_checkpoint_sha256"),
        parity_source_l15_checkpoint_sha256=parity.get("source_l15_checkpoint_sha256"),
        resource_status=str(resources.get("decision", {}).get("status", "")),
        resource_checkpoint_sha256=resources.get("standalone_checkpoint_sha256"),
        resource_source_l15_checkpoint_sha256=resources.get("source_l15_checkpoint_sha256"),
    )
    result = {
        "schema": "NOLANE-L16-STANDALONE-PROMOTION-V1",
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite standalone promotion decision: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "STANDALONE_PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
