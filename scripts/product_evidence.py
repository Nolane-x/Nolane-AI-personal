from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nolane_personal.local_evidence_workbench import (
    finalize_workbench,
    paths_for,
    workbench_status,
)
from nolane_personal.product_evidence_bridge import (
    ProductEvidenceExportPolicy,
    build_product_longitudinal_plan,
    prepare_product_evidence_window,
    verify_product_evidence_window,
)


ROOT = Path(__file__).resolve().parents[1]


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(prog="product-evidence")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-window")
    prepare.add_argument("--db", required=True)
    prepare.add_argument("--workspace", required=True)
    prepare.add_argument("--profile", default=None)
    prepare.add_argument("--language", choices=["vi", "en"], default=None)
    prepare.add_argument("--after-rowid", type=int, default=0)
    prepare.add_argument("--through-rowid", type=int, default=None)
    prepare.add_argument("--session-gap-minutes", type=int, default=45)
    prepare.add_argument("--max-pairs-per-group", type=int, default=4)

    status = sub.add_parser("status")
    status.add_argument("--workspace", required=True)

    review = sub.add_parser("review")
    review.add_argument("--workspace", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--workspace", required=True)
    finalize.add_argument("--allow-undecided", action="store_true")

    verify = sub.add_parser("verify-window")
    verify.add_argument("--workspace", required=True)

    plan = sub.add_parser("build-plan")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "prepare-window":
        policy = ProductEvidenceExportPolicy(
            session_gap_minutes=args.session_gap_minutes,
            max_pairs_per_group=args.max_pairs_per_group,
        )
        emit(
            prepare_product_evidence_window(
                args.db,
                args.workspace,
                after_rowid=args.after_rowid,
                through_rowid=args.through_rowid,
                profile_path=args.profile,
                language=args.language,
                policy=policy,
            )
        )
        return 0

    if args.command == "status":
        root = Path(args.workspace)
        emit(
            {
                "window": verify_product_evidence_window(root),
                "workbench": workbench_status(root / "workbench"),
            }
        )
        return 0

    if args.command == "verify-window":
        emit(verify_product_evidence_window(args.workspace))
        return 0

    if args.command == "review":
        root = Path(args.workspace)
        verify_product_evidence_window(root)
        wb = paths_for(root / "workbench")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "review_local_queue.py"),
                "--queue-manifest",
                str(wb.queue_manifest),
                "--decisions",
                str(wb.decisions),
                "--progress-manifest",
                str(wb.review_progress),
            ],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        emit(workbench_status(root / "workbench"))
        return 0

    if args.command == "finalize":
        root = Path(args.workspace)
        verify_product_evidence_window(root)
        emit(
            finalize_workbench(
                root / "workbench",
                allow_undecided=args.allow_undecided,
            )
        )
        return 0

    if args.command == "build-plan":
        emit(build_product_longitudinal_plan(args.spec, args.output))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
