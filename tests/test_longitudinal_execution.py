import json
from pathlib import Path

import pytest

import nolane_personal.longitudinal_execution as le
from nolane_personal.approved_evidence import build_approved_evidence_pack
from nolane_personal.longitudinal_execution import (
    PLAN_SCHEMA,
    build_longitudinal_report,
    validate_longitudinal_plan,
    verify_longitudinal_report_digest,
)


def write_source(path: Path, prefix: str, *, groups=None, count: int = 12):
    prompts = [
        "Explain why a bicycle stays easier to balance while moving forward.",
        "Mình muốn một lời nhắc ngắn để nhớ mang áo mưa trước khi ra ngoài.",
        "Compare a paper map with GPS navigation for a day hike.",
        "Viết một câu trả lời lịch sự khi mình cần từ chối một cuộc hẹn.",
        "Describe how bread dough changes while yeast ferments.",
        "Nếu mình học từ mới, hãy gợi ý cách ôn lại sau ba ngày.",
        "What makes a mechanical keyboard switch feel different from another?",
        "Tóm tắt cách sắp xếp bàn học để ít bị phân tâm hơn.",
        "Explain the difference between a meteor and a meteorite simply.",
        "Mình muốn chuẩn bị đồ cho chuyến đi ngắn hai ngày, nên nhớ gì?",
        "Give one practical way to verify a downloaded file has not changed.",
        "Giải thích vì sao cây trong phòng vẫn cần ánh sáng dù được tưới đủ nước.",
    ]
    targets = [
        "Forward motion makes steering corrections effective, so small tilts can be corrected before they grow.",
        "Trước khi đi, nhớ kiểm tra dự báo và bỏ một chiếc áo mưa gọn vào túi.",
        "A paper map works without power and shows broad context, while GPS is faster for live position and rerouting.",
        "Mình cảm ơn lời mời, nhưng lần này mình không sắp xếp tham gia được. Hẹn dịp khác nhé.",
        "Yeast consumes sugars and releases carbon dioxide, which becomes trapped in the dough and makes it expand.",
        "Sau ba ngày, hãy tự nhớ nghĩa trước, rồi đặt từ đó vào một câu mới thay vì chỉ đọc lại.",
        "Switches differ in spring weight, travel, tactile bumps, sound, and housing construction, which changes the key feel.",
        "Giữ trên bàn chỉ những thứ đang dùng, đặt điện thoại ngoài tầm tay và chừa một vùng trống để viết.",
        "A meteor is the streak of light from material burning in the atmosphere; a meteorite is what reaches the ground.",
        "Ưu tiên giấy tờ, sạc, thuốc cần thiết, quần áo theo thời tiết và một bộ dự phòng gọn nhẹ.",
        "Compute a cryptographic hash such as SHA-256 and compare it with the trusted value published by the source.",
        "Cây cần ánh sáng để quang hợp tạo năng lượng; nước không thể thay thế phần năng lượng đó.",
    ]
    if count > len(prompts):
        raise ValueError("test fixture supports at most 12 evidence rows")
    groups = groups or [f"{prefix}-group-{i}" for i in range(count)]
    assert len(groups) == count
    rows = []
    for i in range(count):
        rows.append(
            {
                "prompt": f"{prefix}: {prompts[i]}",
                "target": targets[i],
                "language": "vi" if i % 2 else "en",
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
    (tokenizer / "tokenizer.json").write_text(
        '{"version":"1.0","model":{"type":"fixture"}}',
        encoding="utf-8",
    )
    (tokenizer / "tokenizer_config.json").write_text(
        '{"chat_template":"fixture"}',
        encoding="utf-8",
    )

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
    assert len(plan.receipt["initial_factorized_sha256"]) == 64
    assert len(plan.receipt["latent_sha256"]) == 64
    assert len(plan.receipt["tokenizer_assets_sha256"]) == 64
    assert plan.receipt["cycle_bindings"][0]["training"] == {
        "epochs": 1,
        "learning_rate": 0.0002,
        "adaptation_task_weight": 1.0,
        "retention_task_weight": 0.5,
        "retention_distill_weight": 1.0,
        "cortex_anchor_weight": 0.01,
        "temperature": 2.0,
        "max_grad_norm": 1.0,
        "max_length": 128,
    }

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


def fake_plan_receipt(cycles):
    return {
        "cycles": len(cycles),
        "plan_sha256": "d" * 64,
        "fixed_panel": {
            "protocol_sha256": "f" * 64,
        },
        "cycle_bindings": [
            {
                "adaptation": {
                    "protocol_sha256": cycle["training"]["lineage"][
                        "adaptation_protocol_sha256"
                    ],
                }
            }
            for cycle in cycles
        ],
    }


def fake_window(i: int, cycles, *, status="PASS"):
    return {
        "status": status,
        "court_sha256": f"{200 + i:064x}",
        "protocol_sha256": cycles[i]["training"]["lineage"][
            "adaptation_protocol_sha256"
        ],
        "initial_checkpoint_sha256": cycles[i]["artifact"][
            "checkpoint_sha256"
        ],
        "final_checkpoint_sha256": cycles[-1]["artifact"][
            "checkpoint_sha256"
        ],
        "group_robustness": {
            "summary": {
                "overall_regression": 0.004,
                "worst_group_regression": 0.009,
            }
        },
    }


def fake_fixed():
    return {
        "status": "PASS",
        "court_sha256": "c" * 64,
        "protocol_sha256": "f" * 64,
    }


def fake_unified(cycles):
    return {
        "status": "PASS",
        "policy": {
            "min_cycles": 5,
            "min_cortex_cycles": 5,
            "require_unique_adaptation_protocols": True,
        },
        "first_parent_checkpoint_sha256": "a" * 64,
        "final_artifact_checkpoint_sha256": cycles[-1]["artifact"][
            "checkpoint_sha256"
        ],
        "chain_sha256": "b" * 64,
    }


def patch_report_primitives(monkeypatch, unified):
    monkeypatch.setattr(
        le,
        "verify_longitudinal_plan_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        le,
        "verify_l38_run_receipt",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        le,
        "verify_long_horizon_retention_digest",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        le,
        "verify_unified_continual_chain_digest",
        lambda receipt: receipt,
    )
    monkeypatch.setattr(
        le,
        "assess_unified_continual_chain",
        lambda cycles, long_horizon_retention, policy: unified,
    )


def test_longitudinal_report_requires_every_learned_window_to_survive(
    monkeypatch,
):
    cycles = [fake_cycle(i) for i in range(5)]
    windows = [fake_window(i, cycles) for i in range(4)]
    unified = fake_unified(cycles)
    fixed = fake_fixed()
    plan_receipt = fake_plan_receipt(cycles)
    patch_report_primitives(monkeypatch, unified)

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
    assert report["cycle_rows"][-1]["future_cycles_observed"] == 0
    assert report["cycle_rows"][-1]["learned_window_court_sha256"] is None
    verify_longitudinal_report_digest(report)

    windows[1] = fake_window(1, cycles, status="BLOCKED")
    blocked = build_longitudinal_report(
        plan_receipt=plan_receipt,
        cycle_receipts=cycles,
        fixed_panel_receipt=fixed,
        learned_window_receipts=windows,
        unified_chain=unified,
    )
    assert blocked["status"] == "BLOCKED"
    assert "learned_window_002_forgotten" in blocked["reasons"]


def test_longitudinal_report_digest_tamper_is_detected(monkeypatch):
    cycles = [fake_cycle(i) for i in range(5)]
    unified = fake_unified(cycles)
    patch_report_primitives(monkeypatch, unified)
    report = build_longitudinal_report(
        plan_receipt=fake_plan_receipt(cycles),
        cycle_receipts=cycles,
        fixed_panel_receipt=fake_fixed(),
        learned_window_receipts=[
            fake_window(i, cycles) for i in range(4)
        ],
        unified_chain=unified,
    )
    report["cycles"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_longitudinal_report_digest(report)


def test_longitudinal_report_rejects_recomputed_chain_mismatch(monkeypatch):
    cycles = [fake_cycle(i) for i in range(5)]
    unified = fake_unified(cycles)
    patch_report_primitives(monkeypatch, unified)
    monkeypatch.setattr(
        le,
        "assess_unified_continual_chain",
        lambda cycles, long_horizon_retention, policy: {
            **unified,
            "chain_sha256": "e" * 64,
        },
    )
    with pytest.raises(ValueError, match="does not match longitudinal raw evidence"):
        build_longitudinal_report(
            plan_receipt=fake_plan_receipt(cycles),
            cycle_receipts=cycles,
            fixed_panel_receipt=fake_fixed(),
            learned_window_receipts=[
                fake_window(i, cycles) for i in range(4)
            ],
            unified_chain=unified,
        )


def test_longitudinal_report_requires_exactly_cycles_minus_one_future_courts(
    monkeypatch,
):
    cycles = [fake_cycle(i) for i in range(5)]
    unified = fake_unified(cycles)
    patch_report_primitives(monkeypatch, unified)
    with pytest.raises(ValueError, match="cycles - 1"):
        build_longitudinal_report(
            plan_receipt=fake_plan_receipt(cycles),
            cycle_receipts=cycles,
            fixed_panel_receipt=fake_fixed(),
            learned_window_receipts=[
                fake_window(i % 4, cycles) for i in range(5)
            ],
            unified_chain=unified,
        )


def test_longitudinal_policy_rejects_looser_evidence_thresholds(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["policy"] = {
        "max_fixed_overall_regression": 0.02,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be looser"):
        validate_longitudinal_plan(path)


def test_longitudinal_policy_cannot_disable_fixed_panel_isolation(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["policy"] = {
        "require_fixed_panel_isolation": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be disabled"):
        validate_longitudinal_plan(path)


def test_longitudinal_plan_rejects_unknown_training_fields(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    payload["cycles"][0]["training"]["mystery_knob"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported longitudinal training fields"):
        validate_longitudinal_plan(path)


def test_longitudinal_plan_identity_changes_when_tokenizer_changes(tmp_path):
    path, payload = make_plan(tmp_path, cycles=5)
    first = validate_longitudinal_plan(path)
    tokenizer = Path(payload["tokenizer"])
    (tokenizer / "tokenizer.json").write_text(
        '{"version":"2.0","model":{"type":"fixture"}}',
        encoding="utf-8",
    )
    second = validate_longitudinal_plan(path)
    assert (
        first.receipt["tokenizer_assets_sha256"]
        != second.receipt["tokenizer_assets_sha256"]
    )
    assert first.receipt["plan_sha256"] != second.receipt["plan_sha256"]
