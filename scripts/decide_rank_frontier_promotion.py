from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.rank_frontier_promotion import decide_rank_frontier_promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frontier = json.loads(Path(args.frontier).read_text(encoding="utf-8"))
    quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
    resources = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    decision = decide_rank_frontier_promotion(
        frontier_selected_rank=frontier.get("selected_rank"),
        frontier_checkpoint_sha256=frontier.get("selected_checkpoint_sha256"),
        frontier_source_l16_checkpoint_sha256=frontier.get("source_l16_checkpoint_sha256"),
        quality_status=str(quality.get("decision", {}).get("status", "")),
        quality_rank=quality.get("decision", {}).get("evidence", {}).get("rank"),
        quality_checkpoint_sha256=quality.get("factorized_checkpoint_sha256"),
        quality_source_l16_checkpoint_sha256=quality.get("source_l16_checkpoint_sha256"),
        resource_status=str(resources.get("decision", {}).get("status", "")),
        resource_checkpoint_sha256=resources.get("factorized_checkpoint_sha256"),
        resource_source_l16_checkpoint_sha256=resources.get("source_l16_checkpoint_sha256"),
    )
    result = {
        "schema": "NOLANE-L18-RANK-FRONTIER-PROMOTION-V1",
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
    return 0 if decision.status == "ADAPTIVE_RANK_FRONTIER_PROMOTION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
