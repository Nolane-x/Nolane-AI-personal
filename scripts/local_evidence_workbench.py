from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nolane_personal.local_evidence_workbench import (
    assess_workbench_readiness,
    finalize_workbench,
    initialize_workbench,
    paths_for,
    verify_workbench_manifest,
    workbench_status,
)


ROOT = Path(__file__).resolve().parents[1]


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--source", required=True)
    init.add_argument("--workspace", default="runtime-data/local-evidence-workbench")

    status = sub.add_parser("status")
    status.add_argument("--workspace", default="runtime-data/local-evidence-workbench")

    review = sub.add_parser("review")
    review.add_argument("--workspace", default="runtime-data/local-evidence-workbench")

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--workspace", default="runtime-data/local-evidence-workbench")
    finalize.add_argument("--allow-undecided", action="store_true")

    verify = sub.add_parser("verify")
    verify.add_argument("--workspace", default="runtime-data/local-evidence-workbench")

    readiness = sub.add_parser("readiness")
    readiness.add_argument("--workspace", default="runtime-data/local-evidence-workbench")
    readiness.add_argument("--anchor", default=str(ROOT/"research/personalization-general-anchor.jsonl"))
    readiness.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    readiness.add_argument("--model-lock", default=str(ROOT/"model.lock.json"))
    readiness.add_argument("--model", default=str(ROOT/"models/Qwen3-0.6B"))
    readiness.add_argument("--l14-anchor", default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt")

    args = parser.parse_args()

    if args.command == "init":
        emit(initialize_workbench(args.source, args.workspace))
        return 0
    if args.command == "status":
        emit(workbench_status(args.workspace))
        return 0
    if args.command == "verify":
        emit(verify_workbench_manifest(args.workspace))
        return 0
    if args.command == "finalize":
        emit(finalize_workbench(
            args.workspace,
            allow_undecided=args.allow_undecided,
        ))
        return 0
    if args.command == "readiness":
        result=assess_workbench_readiness(
            args.workspace,
            anchor=args.anchor,
            latent=args.latent,
            model_lock=args.model_lock,
            model_dir=args.model,
            l14_anchor=args.l14_anchor,
        )
        emit(result)
        return 0 if result["readiness"]["status"]=="REAL_CANDIDATE_INPUTS_READY" else 2
    if args.command == "review":
        paths=paths_for(args.workspace)
        completed=subprocess.run(
            [
                sys.executable,
                str(ROOT/"scripts/review_local_queue.py"),
                "--queue-manifest",str(paths.queue_manifest),
                "--decisions",str(paths.decisions),
                "--progress-manifest",str(paths.review_progress),
            ],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        emit(workbench_status(args.workspace))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
