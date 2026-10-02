import json
import subprocess
import sys
from pathlib import Path

import pytest

from nolane_personal.evidence_chain import (
    EvidenceStageError,
    REAL_CANDIDATE_STAGE_ORDER,
    evidence_chain_contract_sha256,
    run_stage,
    verify_complete_stage_order,
    write_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_chain_contract_is_exact_and_stable():
    assert len(REAL_CANDIDATE_STAGE_ORDER) == 17
    assert len(set(REAL_CANDIDATE_STAGE_ORDER)) == 17
    assert REAL_CANDIDATE_STAGE_ORDER[0] == "freeze_l15_spec"
    assert REAL_CANDIDATE_STAGE_ORDER[-1] == "promote_l19"
    digest = evidence_chain_contract_sha256()
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == evidence_chain_contract_sha256()


def test_synthetic_fixture_executes_all_17_stages_without_private_text_leak(tmp_path):
    workspace = tmp_path / "fixture"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_evidence_chain_fixture.py"),
            "--workspace",
            str(workspace),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    receipt_path = workspace / "fixture-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["authority"] == "SYNTHETIC_NON_AUTHORITY_NEVER_PROMOTABLE"
    assert receipt["status"] == "SYNTHETIC_EVIDENCE_CHAIN_PASS"
    assert receipt["blocked_stage"] is None
    assert receipt["stage_count_expected"] == 17
    assert receipt["stage_count_observed"] == 17
    assert receipt["stage_contract_sha256"] == evidence_chain_contract_sha256()
    assert [row["name"] for row in receipt["stages"]] == list(
        REAL_CANDIDATE_STAGE_ORDER
    )
    for index, row in enumerate(receipt["stages"], start=1):
        assert row["stage_index"] == index
        assert row["exit_code"] == 0
        assert len(row["expected_outputs"]) == 1
        assert len(row["output_sha256"]) == 1

    rendered = receipt_path.read_text(encoding="utf-8")
    assert "L22-SYNTHETIC-PRIVATE-PROMPT" not in rendered
    assert "L22-SYNTHETIC-PRIVATE-TARGET" not in rendered


def test_synthetic_fixture_stops_at_first_failed_stage(tmp_path):
    workspace = tmp_path / "fixture-fail"
    fail_stage = "evaluate_l16_parity"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_evidence_chain_fixture.py"),
            "--workspace",
            str(workspace),
            "--fail-stage",
            fail_stage,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 7
    receipt = json.loads(
        (workspace / "fixture-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "SYNTHETIC_EVIDENCE_CHAIN_BLOCKED"
    assert receipt["blocked_stage"] == fail_stage
    observed = [row["name"] for row in receipt["stages"]]
    failed_index = REAL_CANDIDATE_STAGE_ORDER.index(fail_stage)
    assert observed == list(REAL_CANDIDATE_STAGE_ORDER[: failed_index + 1])
    assert receipt["stages"][-1]["exit_code"] == 7
    assert not (
        workspace
        / "artifacts"
        / f"{failed_index + 2:02d}-{REAL_CANDIDATE_STAGE_ORDER[failed_index + 1]}.json"
    ).exists()


def test_shared_engine_fails_closed_on_out_of_order_stage(tmp_path):
    receipt = {
        "status": "RUNNING",
        "blocked_stage": None,
        "stages": [],
    }
    receipt_path = tmp_path / "receipt.json"

    def runner(_command, _cwd):
        return 0

    with pytest.raises(EvidenceStageError, match="stage_order_mismatch"):
        run_stage(
            name="train_l15_native",
            command=["synthetic"],
            expected=[tmp_path / "never-created.json"],
            receipt=receipt,
            receipt_path=receipt_path,
            root=ROOT,
            runner=runner,
        )
    assert receipt["blocked_stage"] == "train_l15_native"
    assert receipt["stages"] == []


def test_complete_order_requires_digest_coverage_for_every_output(tmp_path):
    receipt = {"stages": []}
    for index, name in enumerate(REAL_CANDIDATE_STAGE_ORDER, start=1):
        path = tmp_path / f"{index}.json"
        path.write_text("{}", encoding="utf-8")
        receipt["stages"].append(
            {
                "stage_index": index,
                "name": name,
                "exit_code": 0,
                "expected_outputs": [str(path)],
                "output_sha256": {str(path): "a" * 64},
            }
        )
    verify_complete_stage_order(receipt)

    receipt["stages"][-1]["output_sha256"] = {}
    with pytest.raises(ValueError, match="output digest coverage mismatch"):
        verify_complete_stage_order(receipt)
