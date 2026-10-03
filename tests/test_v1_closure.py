from __future__ import annotations

import json

from nolane_personal.longitudinal_execution import (
    AUTHORITY as L43_AUTHORITY,
    REPORT_SCHEMA as L43_REPORT_SCHEMA,
)
from nolane_personal.promotion_ceremony import (
    AUTHORITY as L36_AUTHORITY,
    SCHEMA as L36_SCHEMA,
)
from nolane_personal.store import payload_digest
from nolane_personal.v1_closure import (
    BLOCKED_STATUS,
    DEVICE_SCHEMA,
    PERFORMANCE_POLICY_SCHEMA,
    READY_STATUS,
    CI_READY_STATUS,
    build_android_device_receipt,
    build_android_performance_receipt,
    build_ci_v1_closure_receipt,
    evaluate_v1_closure,
    verify_ci_v1_closure_receipt,
    verify_v1_closure_receipt,
)


VERSION = "0.60.0"
CHECKPOINT = "a" * 64
CEREMONY_AUTH = "b" * 64
DEVICE_FINGERPRINT = "d" * 64
APK_SHA = "e" * 64


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def longitudinal_report(checkpoint=CHECKPOINT):
    body = {
        "schema": L43_REPORT_SCHEMA,
        "authority": L43_AUTHORITY,
        "status": "PASS",
        "reasons": [],
        "plan_sha256": "1" * 64,
        "cycles": 5,
        "initial_checkpoint_sha256": "2" * 64,
        "final_checkpoint_sha256": checkpoint,
        "fixed_panel_court_sha256": "3" * 64,
        "unified_chain_sha256": "4" * 64,
        "cycle_rows": [],
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_source_group_hash_values": False,
            "contains_local_paths": False,
        },
    }
    body["report_sha256"] = payload_digest(body)
    return body


def promotion_ceremony(
    checkpoint=CHECKPOINT,
    transaction_id="real-v1-release-001",
):
    body = {
        "schema": L36_SCHEMA,
        "authority": L36_AUTHORITY,
        "status": "COMPLETE",
        "reasons": [],
        "authorization_sha256": CEREMONY_AUTH,
        "multicycle_chain_sha256": "5" * 64,
        "long_horizon_retention_court_sha256": "6" * 64,
        "candidate_checkpoint_sha256": checkpoint,
        "pointer_sha256": "7" * 64,
        "serving_convergence_sha256": "8" * 64,
        "transaction_id": transaction_id,
        "pointer_generation": 7,
        "authorization_issued_at": "2026-10-01T00:00:00+00:00",
        "authorization_expires_at": "2027-10-01T00:00:00+00:00",
        "transaction_prepared_at": "2026-10-01T00:01:00+00:00",
        "transaction_committed_at": "2026-10-01T00:02:00+00:00",
        "pointer_created_at": "2026-10-01T00:02:00+00:00",
        "serving_convergence_assessed_at": "2026-10-01T00:03:00+00:00",
        "ceremony_at": "2026-10-01T00:04:00+00:00",
    }
    body["ceremony_sha256"] = payload_digest(body)
    return body


def windows_receipt(ceremony, checkpoint=CHECKPOINT):
    return {
        "schema": "NOLANE-V049-WINDOWS-CLEAN-INSTALL-RECEIPT-V1",
        "status": "PASS",
        "installer_sha256": "9" * 64,
        "installed_app_sha256": "a" * 64,
        "installed_runtime_sha256": "b" * 64,
        "installed_model_sha256": checkpoint,
        "installed_tokenizer_json_sha256": "c" * 64,
        "installed_tokenizer_config_sha256": "d" * 64,
        "promotion_ceremony_sha256": ceremony["ceremony_sha256"],
        "release_manifest_model_sha256": checkpoint,
        "product_version": VERSION,
        "app_file_version": VERSION + ".0",
        "readiness_status": "PASS",
        "readiness_critical_failures": 0,
        "ai_phase": "on",
        "chat_reply_nonempty": True,
        "history_messages": 2,
        "installed_app_spawned_runtime": True,
        "privacy": {
            "contains_chat_text": False,
            "contains_auth_token": False,
            "contains_user_data_path": False,
        },
    }


def performance_policy():
    return {
        "schema": PERFORMANCE_POLICY_SCHEMA,
        "minimum_samples": 5,
        "max_cold_boot_ms": 30000,
        "max_p95_turn_ms": 30000,
        "max_peak_pss_mb": 2048,
        "max_battery_drain_pct_per_hour": 25.0,
        "max_thermal_status": 3,
        "max_crash_count": 0,
    }


def device_receipt(ceremony, *, evidence_class="REAL_PHYSICAL_DEVICE"):
    return build_android_device_receipt(
        evidence_class=evidence_class,
        product_version=VERSION,
        apk_sha256=APK_SHA,
        candidate_checkpoint_sha256=CHECKPOINT,
        promotion_ceremony_sha256=ceremony["ceremony_sha256"],
        device_fingerprint_sha256=DEVICE_FINGERPRINT,
        android_sdk=35,
        physical_device=True,
        installed_version_pass=True,
        first_boot_pass=True,
        local_chat_pass=True,
        force_stop_restart_pass=True,
        identity_continuity_pass=True,
        history_continuity_pass=True,
    )


def performance_receipt(*, evidence_class="REAL_PHYSICAL_DEVICE", slow=False):
    metrics = {
        "cold_boot_ms": 3500,
        "p95_turn_ms": 35000 if slow else 4200,
        "peak_pss_mb": 780,
        "battery_drain_pct_per_hour": 8.5,
        "max_thermal_status": 2,
        "crash_count": 0,
    }
    return build_android_performance_receipt(
        evidence_class=evidence_class,
        apk_sha256=APK_SHA,
        candidate_checkpoint_sha256=CHECKPOINT,
        device_fingerprint_sha256=DEVICE_FINGERPRINT,
        policy=performance_policy(),
        sample_count=7,
        metrics=metrics,
    )


def full_evidence(tmp_path, *, ceremony=None, report=None, device=None, performance=None):
    ceremony = ceremony or promotion_ceremony()
    report = report or longitudinal_report()
    device = device or device_receipt(ceremony)
    performance = performance or performance_receipt()
    return {
        "longitudinal_report_path": write_json(
            tmp_path / "l43-report.json", report
        ),
        "promotion_ceremony_path": write_json(
            tmp_path / "ceremony.json", ceremony
        ),
        "windows_receipt_path": write_json(
            tmp_path / "windows.json",
            windows_receipt(ceremony),
        ),
        "android_device_receipt_path": write_json(
            tmp_path / "android-device.json", device
        ),
        "android_performance_receipt_path": write_json(
            tmp_path / "android-performance.json", performance
        ),
        "android_performance_policy_path": write_json(
            tmp_path / "policy.json", performance_policy()
        ),
        "expected_product_version": VERSION,
    }


def test_v1_closure_blocks_when_real_evidence_is_missing(tmp_path):
    policy = write_json(tmp_path / "policy.json", performance_policy())
    receipt = evaluate_v1_closure(
        longitudinal_report_path=None,
        promotion_ceremony_path=None,
        windows_receipt_path=None,
        android_device_receipt_path=None,
        android_performance_receipt_path=None,
        android_performance_policy_path=policy,
        expected_product_version=VERSION,
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert "missing_longitudinal_report" in receipt["reasons"]
    assert "missing_promotion_ceremony" in receipt["reasons"]
    assert "missing_windows_clean_install" in receipt["reasons"]
    assert "missing_android_physical_device" in receipt["reasons"]
    assert "missing_android_performance" in receipt["reasons"]
    verify_v1_closure_receipt(receipt)


def test_v1_closure_ready_only_when_all_evidence_binds_one_candidate(tmp_path):
    receipt = evaluate_v1_closure(**full_evidence(tmp_path))
    assert receipt["status"] == READY_STATUS
    assert receipt["reasons"] == []
    assert receipt["candidate_checkpoint_sha256"] == CHECKPOINT
    assert receipt["privacy"]["contains_local_paths"] is False
    verify_v1_closure_receipt(receipt)


def test_v1_closure_blocks_synthetic_promotion_even_if_mechanics_pass(tmp_path):
    ceremony = promotion_ceremony(
        transaction_id="synthetic-mobile-release-court"
    )
    receipt = evaluate_v1_closure(
        **full_evidence(
            tmp_path,
            ceremony=ceremony,
            device=device_receipt(ceremony),
        )
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert "synthetic_promotion_ceremony" in receipt["reasons"]


def test_v1_closure_blocks_checkpoint_mixing(tmp_path):
    receipt = evaluate_v1_closure(
        **full_evidence(
            tmp_path,
            report=longitudinal_report("f" * 64),
        )
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert "l43_final_checkpoint_not_promoted" in receipt["reasons"]


def test_v1_closure_blocks_synthetic_device_evidence(tmp_path):
    ceremony = promotion_ceremony()
    synthetic = device_receipt(
        ceremony,
        evidence_class="SYNTHETIC_COURT",
    )
    receipt = evaluate_v1_closure(
        **full_evidence(
            tmp_path,
            ceremony=ceremony,
            device=synthetic,
        )
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        reason.startswith("invalid_android_physical_device")
        for reason in receipt["reasons"]
    )


def test_v1_closure_blocks_resource_policy_failure(tmp_path):
    receipt = evaluate_v1_closure(
        **full_evidence(
            tmp_path,
            performance=performance_receipt(slow=True),
        )
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert "android_performance_not_pass" in receipt["reasons"]

def test_v1_closure_rejects_loosened_performance_policy(tmp_path):
    evidence = full_evidence(tmp_path)
    loosened = performance_policy()
    loosened["max_p95_turn_ms"] = 60000
    write_json(evidence["android_performance_policy_path"], loosened)
    receipt = evaluate_v1_closure(**evidence)
    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        reason.startswith("invalid_android_performance_policy")
        for reason in receipt["reasons"]
    )


def test_v1_closure_rejects_windows_manifest_model_drift(tmp_path):
    ceremony = promotion_ceremony()
    evidence = full_evidence(tmp_path, ceremony=ceremony)
    windows = windows_receipt(ceremony)
    windows["release_manifest_model_sha256"] = "f" * 64
    write_json(evidence["windows_receipt_path"], windows)
    receipt = evaluate_v1_closure(**evidence)
    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        reason.startswith("invalid_windows_clean_install")
        for reason in receipt["reasons"]
    )

def successful_ci_workflows():
    return {
        "Product Client Court": "success",
        "Living Runtime CI": "success",
        "Neural Shadow CI": "success",
        "Platform Crash Court": "success",
    }


def successful_product_jobs():
    return {
        "Product runtime court": "success",
        "Android APK court": "success",
        "NUI browser court": "success",
        "Android native kernel": "success",
        "Windows native host": "success",
        "Android x86_64 emulator APK": "success",
        "Android emulator local-chat restart court": "success",
    }


def test_ci_v1_closure_ready_on_main_when_all_required_courts_pass():
    receipt = build_ci_v1_closure_receipt(
        repository="Nolane-x/Nolane-AI-personal",
        branch="main",
        commit_sha="1" * 40,
        product_version=VERSION,
        workflows=successful_ci_workflows(),
        product_jobs=successful_product_jobs(),
    )
    assert receipt["status"] == CI_READY_STATUS
    assert receipt["reasons"] == []
    assert "physical_device_certification" in receipt["excluded_claims"]
    assert "field_battery_runtime" in receipt["excluded_claims"]
    verify_ci_v1_closure_receipt(receipt)


def test_ci_v1_closure_blocks_off_main():
    receipt = build_ci_v1_closure_receipt(
        repository="Nolane-x/Nolane-AI-personal",
        branch="feature",
        commit_sha="2" * 40,
        product_version=VERSION,
        workflows=successful_ci_workflows(),
        product_jobs=successful_product_jobs(),
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert "not_main_branch" in receipt["reasons"]


def test_ci_v1_closure_blocks_any_required_workflow_failure():
    workflows = successful_ci_workflows()
    workflows["Neural Shadow CI"] = "failure"
    receipt = build_ci_v1_closure_receipt(
        repository="Nolane-x/Nolane-AI-personal",
        branch="main",
        commit_sha="3" * 40,
        product_version=VERSION,
        workflows=workflows,
        product_jobs=successful_product_jobs(),
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        reason.startswith("workflow_not_success:Neural Shadow CI")
        for reason in receipt["reasons"]
    )


def test_ci_v1_closure_blocks_any_required_packaging_job_failure():
    jobs = successful_product_jobs()
    jobs["Android emulator local-chat restart court"] = "failure"
    receipt = build_ci_v1_closure_receipt(
        repository="Nolane-x/Nolane-AI-personal",
        branch="main",
        commit_sha="4" * 40,
        product_version=VERSION,
        workflows=successful_ci_workflows(),
        product_jobs=jobs,
    )
    assert receipt["status"] == BLOCKED_STATUS
    assert any(
        reason.startswith(
            "product_job_not_success:Android emulator local-chat restart court"
        )
        for reason in receipt["reasons"]
    )

