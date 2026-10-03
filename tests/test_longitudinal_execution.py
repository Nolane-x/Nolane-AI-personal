import json
from pathlib import Path

import pytest

from nolane_personal.approved_evidence import build_approved_evidence_pack
from nolane_personal.longitudinal_execution import (
    PLAN_SCHEMA,
    build_longitudinal_report,
    validate_longitudinal_plan,
    verify_longitudinal_report_digest,
)


def write_source(path: Path, prefix: str, *, groups=None, count: int = 12):
    rows = []
    groups = groups or [f"{prefix}-group-{i}" for i in range(count)]
    assert len(groups) == count
    for i in range(count):
        rows.append(
            {
                "prompt": f"{prefix} prompt {i} with enough distinction",
                "target": f"{prefix} target {i} with a concrete answer",
                "language": "vi" if i % 2 == 0 else "en",
                "weight": 1.0,
                "approved": True,
                "sensitive": False,
                "source_id": groups[i],
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def pack(tmp_path: Path, name: str, *, groups=None):
    source = tmp_path / f"{name}.jsonl"
    write_source(source, name, groups=groups)
    result = build_approved_evidence_pack(
        source,
        tmp_path / f"{name}-pack",
    )
    return result.manifest_path


def make_plan(tmp_path: Path, *, cycles=5):
    initial = tmp_path / "factorized-nolane.pt"
    initial.write_bytes(b"checkpoint")
    latent = tmp_path / "latent.json"
    latent.write_text('{"values":[0.0]}', encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()

    fixed = pack(tmp_path, "fixed")
    cycle_rows = []
    for i in range(cycles):
        cycle_rows.append(
            {
                "retention_manifest": str(pack(tmp_path, f"ret-{i}")),
                "adaptation_manifest": str(pack(tmp_path, f"adapt-{i}")),
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.0002,
                    "max_length": 128,
                },
            }
        )
    payload = {
        "schema": PLAN_SCHEMA,
        "initial_factorized": str(initial),
        "latent": str(latent),
        "tokenizer": str(tokenizer),
        "device": "cpu",
        "fixed_panel_manifest": str(fixed),
        "cycles": cycle_rows,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_real_longitudinal_plan_requires_five_isolated_approved_cycles(tmp_path):
    path, _ = make_plan(tmp_path, cycles=5)
    plan = validate_longitudinal_plan(path)
    assert len(plan.cycles) == 5
    assert plan.receipt["cycles"] == 5
    assert plan.receipt["privacy"]["contains_raw_prompt_target"] is False
    assert plan.receipt["privacy"]["contains_raw_source_group_hash_values"] is False
    assert plan.receipt["privacy"]["contains_local_paths"] is False

    rendered = json.dumps(plan.receipt, sort_keys=True)
    for group in plan.fixed_panel.source_groups:
        assert group not in rendered
    assert str(tmp_path) not in rendered


def test_real_longitudinal_plan_rejects_fewer_than_five_cycles(tmp_path):
    path, _ = make_plan(tmp_path, cycles=4)
    with pytest.raises(ValueError, match="at least 5"):
        validate_longitudinal_plan(path)


def test_real_longitudinal_plan_rejects_reused_adaptation_pack(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["cycles"][1]["adaptation_manifest"] = payload["cycles"][0][
        "adaptation_manifest"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="adaptation protocol reused"):
        validate_longitudinal_plan(path)


def test_real_longitudinal_plan_rejects_retention_adaptation_overlap(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["cycles"][0]["retention_manifest"] = payload["cycles"][0][
        "adaptation_manifest"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="retention/adaptation source groups overlap"):
        validate_longitudinal_plan(path)


def test_real_longitudinal_plan_rejects_fixed_panel_training_overlap(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["cycles"][0]["retention_manifest"] = payload[
        "fixed_panel_manifest"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed long-horizon panel overlaps"):
        validate_longitudinal_plan(path)


def fake_cycle(i: int):
    checkpoint = f"{i + 1:064x}"
    return {
        "training": {
            "lineage": {
                "adaptation_protocol_sha256": f"{100 + i:064x}",
            },
            "continual_learning": {
                "adaptation": {
                    "summary": {
                        "mean_group_gain": 0.2,
                    }
                },
                "retention": {
                    "summary": {
                        "worst_group_regression": 0.005,
                    }
                },
            },
        },
        "artifact": {
            "checkpoint_sha256": checkpoint,
        },
    }


def fake_window(i: int, *, status="PASS"):
    return {
        "status": status,
        "court_sha256": f"{200 + i:064x}",
        "group_robustness": {
            "summary": {
                "overall_regression": 0.004,
                "worst_group_regression": 0.009,
            }
        },
    }


def test_longitudinal_report_requires_every_learned_window_to_survive():
    cycles = [fake_cycle(i) for i in range(5)]
    windows = [fake_window(i) for i in range(5)]
    unified = {
        "status": "PASS",
        "first_parent_checkpoint_sha256": "a" * 64,
        "final_artifact_checkpoint_sha256": cycles[-1]["artifact"][
            "checkpoint_sha256"
        ],
        "chain_sha256": "b" * 64,
    }
    fixed = {
        "status": "PASS",
        "court_sha256": "c" * 64,
    }
    plan_receipt = {"plan_sha256": "d" * 64}

    report = build_longitudinal_report(
        plan_receipt=plan_receipt,
        cycle_receipts=cycles,
        fixed_panel_receipt=fixed,
        learned_window_receipts=windows,
        unified_chain=unified,
    )
    assert report["status"] == "PASS"
    assert report["cycles"] == 5
    assert report["cycle_rows"][0][
        "final_learned_window_overall_regression"
    ] == 0.004
    verify_longitudinal_report_digest(report)

    windows[1] = fake_window(1, status="BLOCKED")
    blocked = build_longitudinal_report(
        plan_receipt=plan_receipt,
        cycle_receipts=cycles,
        fixed_panel_receipt=fixed,
        learned_window_receipts=windows,
        unified_chain=unified,
    )
    assert blocked["status"] == "BLOCKED"
    assert "learned_window_002_forgotten" in blocked["reasons"]


def test_longitudinal_report_digest_tamper_is_detected():
    cycles = [fake_cycle(i) for i in range(5)]
    report = build_longitudinal_report(
        plan_receipt={"plan_sha256": "d" * 64},
        cycle_receipts=cycles,
        fixed_panel_receipt={"status": "PASS", "court_sha256": "c" * 64},
        learned_window_receipts=[fake_window(i) for i in range(5)],
        unified_chain={
            "status": "PASS",
            "first_parent_checkpoint_sha256": "a" * 64,
            "final_artifact_checkpoint_sha256": cycles[-1]["artifact"][
                "checkpoint_sha256"
            ],
            "chain_sha256": "b" * 64,
        },
    )
    report["cycles"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_longitudinal_report_digest(report)
