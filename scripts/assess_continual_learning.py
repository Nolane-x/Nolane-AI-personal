from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
)


EVIDENCE_SCHEMA = "NOLANE-L30-CONTINUAL-EVIDENCE-V1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-retention-groups", type=int, default=2)
    parser.add_argument("--min-adaptation-groups", type=int, default=2)
    parser.add_argument(
        "--max-worst-retention-regression",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-mean-retention-regression",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--min-mean-adaptation-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-worst-adaptation-regression",
        type=float,
        default=0.03,
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    if payload.get("schema") != EVIDENCE_SCHEMA:
        raise SystemExit("unsupported continual-learning evidence schema")

    retention = payload.get("retention")
    adaptation = payload.get("adaptation")
    if not isinstance(retention, dict) or not isinstance(adaptation, dict):
        raise SystemExit("retention/adaptation evidence sections are required")

    policy = ContinualLearningPolicy(
        min_retention_groups=args.min_retention_groups,
        min_adaptation_groups=args.min_adaptation_groups,
        max_worst_retention_regression=args.max_worst_retention_regression,
        max_mean_retention_regression=args.max_mean_retention_regression,
        min_mean_adaptation_gain=args.min_mean_adaptation_gain,
        max_worst_adaptation_regression=args.max_worst_adaptation_regression,
    )
    receipt = assess_continual_learning(
        pre_update_checkpoint_sha256=str(
            payload.get("pre_update_checkpoint_sha256", "")
        ),
        post_update_checkpoint_sha256=str(
            payload.get("post_update_checkpoint_sha256", "")
        ),
        retention_group_sha256=list(retention.get("group_sha256", [])),
        retention_before_values=[
            float(v) for v in retention.get("before_values", [])
        ],
        retention_after_values=[
            float(v) for v in retention.get("after_values", [])
        ],
        adaptation_group_sha256=list(adaptation.get("group_sha256", [])),
        adaptation_before_values=[
            float(v) for v in adaptation.get("before_values", [])
        ],
        adaptation_after_values=[
            float(v) for v in adaptation.get("after_values", [])
        ],
        policy=policy,
    )

    rendered = json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise SystemExit(
                f"refusing to overwrite continual-learning receipt: {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
