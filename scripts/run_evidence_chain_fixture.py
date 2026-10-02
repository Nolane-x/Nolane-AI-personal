from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nolane_personal.evidence_chain import (
    EvidenceStageError,
    REAL_CANDIDATE_STAGE_ORDER,
    evidence_chain_contract_sha256,
    run_stage,
    verify_complete_stage_order,
    write_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="runtime-data/l22-synthetic-evidence-chain",
    )
    parser.add_argument("--fail-stage", default=None)
    args = parser.parse_args()

    if args.fail_stage is not None and args.fail_stage not in REAL_CANDIDATE_STAGE_ORDER:
        raise SystemExit(f"unknown fail stage: {args.fail_stage}")

    workspace = Path(args.workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(
            f"refusing to mix fixture evidence in non-empty workspace: {workspace}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    secret = workspace / "synthetic-private-input.txt"
    secret.write_text(
        "L22-SYNTHETIC-PRIVATE-PROMPT::never-copy-this-into-receipt\n"
        "L22-SYNTHETIC-PRIVATE-TARGET::never-copy-this-into-receipt\n",
        encoding="utf-8",
    )

    receipt_path = workspace / "fixture-receipt.json"
    receipt = {
        "schema": "NOLANE-L22-E2E-EVIDENCE-HARNESS-V1",
        "authority": "SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE",
        "status": "SYNTHETIC_EVIDENCE_CHAIN_RUNNING",
        "blocked_stage": None,
        "stage_contract_sha256": evidence_chain_contract_sha256(),
        "stage_count_expected": len(REAL_CANDIDATE_STAGE_ORDER),
        "stages": [],
    }
    write_receipt(receipt_path, receipt)

    py = sys.executable
    try:
        for index, name in enumerate(REAL_CANDIDATE_STAGE_ORDER, start=1):
            artifact = workspace / "artifacts" / f"{index:02d}-{name}.json"
            command = [
                py,
                "scripts/evidence_fixture_stage.py",
                "--stage",
                name,
                "--secret-file",
                str(secret),
                "--output",
                str(artifact),
            ]
            if name == args.fail_stage:
                command.append("--fail")
            run_stage(
                name=name,
                command=command,
                expected=[artifact],
                receipt=receipt,
                receipt_path=receipt_path,
                root=ROOT,
                blocked_status="SYNTHETIC_EVIDENCE_CHAIN_BLOCKED",
            )
    except EvidenceStageError as exc:
        receipt["authority"] = "SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE"
        write_receipt(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return int(exc.exit_code or 3)

    verify_complete_stage_order(receipt)
    receipt["status"] = "SYNTHETIC_EVIDENCE_CHAIN_PASS"
    receipt["blocked_stage"] = None
    receipt["stage_count_observed"] = len(receipt["stages"])
    write_receipt(receipt_path, receipt)

    rendered = receipt_path.read_text(encoding="utf-8")
    forbidden = (
        "L22-SYNTHETIC-PRIVATE-PROMPT",
        "L22-SYNTHETIC-PRIVATE-TARGET",
    )
    if any(value in rendered for value in forbidden):
        raise SystemExit("synthetic privacy sentinel leaked into fixture receipt")

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
