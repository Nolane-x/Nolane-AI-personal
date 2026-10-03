from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .longitudinal_execution import (
    AUTHORITY as L43_AUTHORITY,
    REPORT_SCHEMA as L43_REPORT_SCHEMA,
    verify_longitudinal_report_digest,
)
from .promotion_ceremony import verify_promotion_ceremony_receipt
from .store import payload_digest


DEVICE_SCHEMA = "NOLANE-V060-ANDROID-PHYSICAL-DEVICE-EVIDENCE-V1"
DEVICE_AUTHORITY = "ANDROID_PHYSICAL_DEVICE_EVIDENCE_NO_PROMOTION_AUTHORITY"
PERFORMANCE_POLICY_SCHEMA = "NOLANE-V060-ANDROID-PERFORMANCE-POLICY-V1"
PERFORMANCE_SCHEMA = "NOLANE-V060-ANDROID-PERFORMANCE-EVIDENCE-V1"
PERFORMANCE_AUTHORITY = "ANDROID_PERFORMANCE_EVIDENCE_NO_PROMOTION_AUTHORITY"
CLOSURE_SCHEMA = "NOLANE-V060-V1-CLOSURE-RECEIPT-V1"
CLOSURE_AUTHORITY = "V1_RELEASE_READINESS_EVALUATION_NO_PROMOTION_AUTHORITY"
READY_STATUS = "READY_FOR_V1_0"
BLOCKED_STATUS = "BLOCKED"

FROZEN_ANDROID_PERFORMANCE_POLICY: dict[str, Any] = {
    "schema": PERFORMANCE_POLICY_SCHEMA,
    "minimum_samples": 5,
    "max_cold_boot_ms": 30000,
    "max_p95_turn_ms": 30000,
    "max_peak_pss_mb": 2048,
    "max_battery_drain_pct_per_hour": 25.0,
    "max_thermal_status": 3,
    "max_crash_count": 0,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 hex string") from exc
    return value.lower()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _verify_self_digest(
    receipt: dict[str, Any],
    *,
    digest_field: str,
    label: str,
) -> None:
    supplied = _sha256(receipt.get(digest_field), field=digest_field)
    body = dict(receipt)
    body.pop(digest_field, None)
    if payload_digest(body) != supplied:
        raise ValueError(f"{label} digest mismatch")


def verify_android_device_receipt(
    receipt: dict[str, Any],
    *,
    require_real: bool = False,
) -> dict[str, Any]:
    if receipt.get("schema") != DEVICE_SCHEMA:
        raise ValueError("unsupported Android physical-device receipt schema")
    if receipt.get("authority") != DEVICE_AUTHORITY:
        raise ValueError("Android physical-device receipt authority mismatch")
    _verify_self_digest(
        receipt,
        digest_field="receipt_sha256",
        label="Android physical-device receipt",
    )
    if receipt.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Android physical-device receipt status invalid")
    if not isinstance(receipt.get("reasons"), list):
        raise ValueError("Android physical-device reasons invalid")
    evidence_class = receipt.get("evidence_class")
    if evidence_class not in {"REAL_PHYSICAL_DEVICE", "SYNTHETIC_COURT"}:
        raise ValueError("Android physical-device evidence_class invalid")
    if require_real and evidence_class != "REAL_PHYSICAL_DEVICE":
        raise ValueError("real physical-device evidence required")
    for key in (
        "apk_sha256",
        "candidate_checkpoint_sha256",
        "promotion_ceremony_sha256",
        "device_fingerprint_sha256",
    ):
        _sha256(receipt.get(key), field=key)
    if int(receipt.get("android_sdk", 0)) < 26:
        raise ValueError("Android physical-device SDK must be >= 26")
    checks = (
        "physical_device",
        "installed_version_pass",
        "first_boot_pass",
        "local_chat_pass",
        "force_stop_restart_pass",
        "identity_continuity_pass",
        "history_continuity_pass",
    )
    for key in checks:
        if not isinstance(receipt.get(key), bool):
            raise ValueError(f"Android physical-device {key} must be boolean")
    if receipt.get("status") == "PASS":
        if receipt.get("reasons"):
            raise ValueError("PASS physical-device receipt has reasons")
        if not all(bool(receipt.get(key)) for key in checks):
            raise ValueError("PASS physical-device receipt has failed checks")
    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("Android physical-device privacy block missing")
    if any(bool(privacy.get(key, True)) for key in (
        "contains_device_fingerprint",
        "contains_chat_text",
        "contains_user_data_path",
    )):
        raise ValueError("Android physical-device receipt leaks private data")
    return receipt


def build_android_device_receipt(
    *,
    evidence_class: str,
    product_version: str,
    apk_sha256: str,
    candidate_checkpoint_sha256: str,
    promotion_ceremony_sha256: str,
    device_fingerprint_sha256: str,
    android_sdk: int,
    physical_device: bool,
    installed_version_pass: bool,
    first_boot_pass: bool,
    local_chat_pass: bool,
    force_stop_restart_pass: bool,
    identity_continuity_pass: bool,
    history_continuity_pass: bool,
) -> dict[str, Any]:
    checks = {
        "physical_device": bool(physical_device),
        "installed_version_pass": bool(installed_version_pass),
        "first_boot_pass": bool(first_boot_pass),
        "local_chat_pass": bool(local_chat_pass),
        "force_stop_restart_pass": bool(force_stop_restart_pass),
        "identity_continuity_pass": bool(identity_continuity_pass),
        "history_continuity_pass": bool(history_continuity_pass),
    }
    reasons = [key for key, passed in checks.items() if not passed]
    receipt: dict[str, Any] = {
        "schema": DEVICE_SCHEMA,
        "authority": DEVICE_AUTHORITY,
        "evidence_class": evidence_class,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "product_version": str(product_version),
        "apk_sha256": _sha256(apk_sha256, field="apk_sha256"),
        "candidate_checkpoint_sha256": _sha256(
            candidate_checkpoint_sha256,
            field="candidate_checkpoint_sha256",
        ),
        "promotion_ceremony_sha256": _sha256(
            promotion_ceremony_sha256,
            field="promotion_ceremony_sha256",
        ),
        "device_fingerprint_sha256": _sha256(
            device_fingerprint_sha256,
            field="device_fingerprint_sha256",
        ),
        "android_sdk": int(android_sdk),
        **checks,
        "privacy": {
            "contains_device_fingerprint": False,
            "contains_chat_text": False,
            "contains_user_data_path": False,
        },
    }
    receipt["receipt_sha256"] = payload_digest(receipt)
    return verify_android_device_receipt(receipt)


def verify_android_performance_policy(
    policy: dict[str, Any],
) -> dict[str, Any]:
    if policy.get("schema") != PERFORMANCE_POLICY_SCHEMA:
        raise ValueError("unsupported Android performance policy schema")
    supplied = policy.get("policy_sha256")
    body = dict(policy)
    body.pop("policy_sha256", None)
    if supplied is not None and payload_digest(body) != supplied:
        raise ValueError("Android performance policy digest mismatch")

    # v0.60 freezes one product acceptance floor. A closure caller may not
    # substitute a looser policy and still claim READY_FOR_V1_0.
    if body != FROZEN_ANDROID_PERFORMANCE_POLICY:
        raise ValueError("Android performance policy differs from frozen v1 policy")
    return policy


def assess_android_performance(
    metrics: dict[str, Any],
    policy: dict[str, Any],
    *,
    sample_count: int,
) -> list[str]:
    verify_android_performance_policy(policy)
    reasons: list[str] = []
    if int(sample_count) < int(policy["minimum_samples"]):
        reasons.append("insufficient_samples")
    comparisons = (
        ("cold_boot_ms", "max_cold_boot_ms"),
        ("p95_turn_ms", "max_p95_turn_ms"),
        ("peak_pss_mb", "max_peak_pss_mb"),
        ("battery_drain_pct_per_hour", "max_battery_drain_pct_per_hour"),
        ("max_thermal_status", "max_thermal_status"),
    )
    for metric, limit in comparisons:
        value = float(metrics.get(metric, float("inf")))
        if value < 0 or value > float(policy[limit]):
            reasons.append(f"{metric}_exceeded")
    crash_count = int(metrics.get("crash_count", -1))
    if crash_count < 0 or crash_count > int(policy["max_crash_count"]):
        reasons.append("crash_count_exceeded")
    return reasons


def build_android_performance_receipt(
    *,
    evidence_class: str,
    apk_sha256: str,
    candidate_checkpoint_sha256: str,
    device_fingerprint_sha256: str,
    policy: dict[str, Any],
    sample_count: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    verify_android_performance_policy(policy)
    policy_body = dict(policy)
    policy_body.pop("policy_sha256", None)
    policy_sha256 = payload_digest(policy_body)
    reasons = assess_android_performance(
        metrics,
        policy,
        sample_count=int(sample_count),
    )
    receipt: dict[str, Any] = {
        "schema": PERFORMANCE_SCHEMA,
        "authority": PERFORMANCE_AUTHORITY,
        "evidence_class": evidence_class,
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "apk_sha256": _sha256(apk_sha256, field="apk_sha256"),
        "candidate_checkpoint_sha256": _sha256(
            candidate_checkpoint_sha256,
            field="candidate_checkpoint_sha256",
        ),
        "device_fingerprint_sha256": _sha256(
            device_fingerprint_sha256,
            field="device_fingerprint_sha256",
        ),
        "policy_sha256": policy_sha256,
        "sample_count": int(sample_count),
        "metrics": {
            "cold_boot_ms": float(metrics["cold_boot_ms"]),
            "p95_turn_ms": float(metrics["p95_turn_ms"]),
            "peak_pss_mb": float(metrics["peak_pss_mb"]),
            "battery_drain_pct_per_hour": float(
                metrics["battery_drain_pct_per_hour"]
            ),
            "max_thermal_status": int(metrics["max_thermal_status"]),
            "crash_count": int(metrics["crash_count"]),
        },
        "privacy": {
            "contains_device_fingerprint": False,
            "contains_chat_text": False,
            "contains_user_data_path": False,
        },
    }
    receipt["receipt_sha256"] = payload_digest(receipt)
    return verify_android_performance_receipt(receipt, policy=policy)


def verify_android_performance_receipt(
    receipt: dict[str, Any],
    *,
    policy: dict[str, Any],
    require_real: bool = False,
) -> dict[str, Any]:
    if receipt.get("schema") != PERFORMANCE_SCHEMA:
        raise ValueError("unsupported Android performance receipt schema")
    if receipt.get("authority") != PERFORMANCE_AUTHORITY:
        raise ValueError("Android performance receipt authority mismatch")
    _verify_self_digest(
        receipt,
        digest_field="receipt_sha256",
        label="Android performance receipt",
    )
    evidence_class = receipt.get("evidence_class")
    if evidence_class not in {"REAL_PHYSICAL_DEVICE", "SYNTHETIC_COURT"}:
        raise ValueError("Android performance evidence_class invalid")
    if require_real and evidence_class != "REAL_PHYSICAL_DEVICE":
        raise ValueError("real Android performance evidence required")
    if receipt.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("Android performance receipt status invalid")
    for key in (
        "apk_sha256",
        "candidate_checkpoint_sha256",
        "device_fingerprint_sha256",
        "policy_sha256",
    ):
        _sha256(receipt.get(key), field=key)
    verify_android_performance_policy(policy)
    policy_body = dict(policy)
    policy_body.pop("policy_sha256", None)
    if receipt["policy_sha256"] != payload_digest(policy_body):
        raise ValueError("Android performance policy binding mismatch")
    reasons = assess_android_performance(
        dict(receipt.get("metrics", {})),
        policy,
        sample_count=int(receipt.get("sample_count", 0)),
    )
    expected = "PASS" if not reasons else "FAIL"
    if receipt.get("status") != expected:
        raise ValueError("Android performance status does not match metrics")
    if list(receipt.get("reasons", [])) != reasons:
        raise ValueError("Android performance reasons do not match metrics")
    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("Android performance privacy block missing")
    if any(bool(privacy.get(key, True)) for key in (
        "contains_device_fingerprint",
        "contains_chat_text",
        "contains_user_data_path",
    )):
        raise ValueError("Android performance receipt leaks private data")
    return receipt


def verify_windows_clean_install_receipt(
    receipt: dict[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    if receipt.get("schema") != "NOLANE-V049-WINDOWS-CLEAN-INSTALL-RECEIPT-V1":
        raise ValueError("unsupported Windows clean-install receipt schema")
    if receipt.get("status") != "PASS":
        raise ValueError("Windows clean-install receipt did not PASS")
    for key in (
        "installer_sha256",
        "installed_app_sha256",
        "installed_runtime_sha256",
        "installed_model_sha256",
        "installed_tokenizer_json_sha256",
        "installed_tokenizer_config_sha256",
        "promotion_ceremony_sha256",
        "release_manifest_model_sha256",
    ):
        _sha256(receipt.get(key), field=key)
    if receipt.get("release_manifest_model_sha256") != receipt.get(
        "installed_model_sha256"
    ):
        raise ValueError("Windows release manifest/model binding mismatch")
    if str(receipt.get("product_version")) != str(expected_version):
        raise ValueError("Windows clean-install product version mismatch")
    if str(receipt.get("app_file_version", "")).startswith(str(expected_version)) is False:
        raise ValueError("Windows installed app file version mismatch")
    if receipt.get("readiness_status") != "PASS":
        raise ValueError("Windows installed readiness did not PASS")
    if int(receipt.get("readiness_critical_failures", -1)) != 0:
        raise ValueError("Windows installed readiness has critical failures")
    if receipt.get("ai_phase") != "on":
        raise ValueError("Windows installed AI did not power on")
    if receipt.get("chat_reply_nonempty") is not True:
        raise ValueError("Windows installed chat did not reply")
    if int(receipt.get("history_messages", 0)) < 2:
        raise ValueError("Windows installed history did not persist")
    if receipt.get("installed_app_spawned_runtime") is not True:
        raise ValueError("Windows installed app did not spawn bundled runtime")
    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict) or any(bool(privacy.get(key, True)) for key in (
        "contains_chat_text",
        "contains_auth_token",
        "contains_user_data_path",
    )):
        raise ValueError("Windows clean-install receipt privacy invalid")
    return receipt


def evaluate_v1_closure(
    *,
    longitudinal_report_path: Path | None,
    promotion_ceremony_path: Path | None,
    windows_receipt_path: Path | None,
    android_device_receipt_path: Path | None,
    android_performance_receipt_path: Path | None,
    android_performance_policy_path: Path | None,
    expected_product_version: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}

    def load(label: str, path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            reasons.append(f"missing_{label}")
            return None
        try:
            payload = _load_json(path)
        except Exception as exc:
            reasons.append(f"invalid_{label}:{type(exc).__name__}")
            return None
        evidence[f"{label}_file_sha256"] = _sha256_file(path)
        loaded[label] = payload
        return payload

    report = load("longitudinal_report", longitudinal_report_path)
    ceremony = load("promotion_ceremony", promotion_ceremony_path)
    windows = load("windows_clean_install", windows_receipt_path)
    device = load("android_physical_device", android_device_receipt_path)
    performance = load("android_performance", android_performance_receipt_path)
    policy = load("android_performance_policy", android_performance_policy_path)

    if report is not None:
        try:
            verify_longitudinal_report_digest(report)
            if report.get("authority") != L43_AUTHORITY:
                raise ValueError("L43 report authority mismatch")
            if report.get("schema") != L43_REPORT_SCHEMA:
                raise ValueError("L43 report schema mismatch")
            if report.get("status") != "PASS":
                reasons.append("l43_longitudinal_not_pass")
            if int(report.get("cycles", 0)) < 5:
                reasons.append("l43_insufficient_cycles")
            privacy = report.get("privacy", {})
            if any(bool(privacy.get(key, True)) for key in (
                "contains_raw_prompt_target",
                "contains_raw_source_group_hash_values",
                "contains_local_paths",
            )):
                reasons.append("l43_privacy_not_closed")
            evidence["longitudinal_report_sha256"] = report.get("report_sha256")
        except Exception as exc:
            reasons.append(f"invalid_longitudinal_report:{type(exc).__name__}")

    if ceremony is not None:
        try:
            verify_promotion_ceremony_receipt(ceremony, require_complete=True)
            if str(ceremony.get("transaction_id", "")).startswith("synthetic-"):
                reasons.append("synthetic_promotion_ceremony")
            evidence["promotion_ceremony_sha256"] = ceremony.get(
                "ceremony_sha256"
            )
        except Exception as exc:
            reasons.append(f"invalid_promotion_ceremony:{type(exc).__name__}")

    if windows is not None:
        try:
            verify_windows_clean_install_receipt(
                windows,
                expected_version=expected_product_version,
            )
        except Exception as exc:
            reasons.append(f"invalid_windows_clean_install:{type(exc).__name__}")

    if device is not None:
        try:
            verify_android_device_receipt(device, require_real=True)
            if device.get("status") != "PASS":
                reasons.append("android_physical_device_not_pass")
        except Exception as exc:
            reasons.append(f"invalid_android_physical_device:{type(exc).__name__}")

    if policy is not None:
        try:
            verify_android_performance_policy(policy)
            policy_body = dict(policy)
            policy_body.pop("policy_sha256", None)
            evidence["android_performance_policy_sha256"] = payload_digest(
                policy_body
            )
        except Exception as exc:
            reasons.append(f"invalid_android_performance_policy:{type(exc).__name__}")

    if performance is not None and policy is not None:
        try:
            verify_android_performance_receipt(
                performance,
                policy=policy,
                require_real=True,
            )
            if performance.get("status") != "PASS":
                reasons.append("android_performance_not_pass")
        except Exception as exc:
            reasons.append(f"invalid_android_performance:{type(exc).__name__}")

    if report is not None and ceremony is not None:
        if report.get("final_checkpoint_sha256") != ceremony.get(
            "candidate_checkpoint_sha256"
        ):
            reasons.append("l43_final_checkpoint_not_promoted")

    if windows is not None and ceremony is not None:
        if windows.get("installed_model_sha256") != ceremony.get(
            "candidate_checkpoint_sha256"
        ):
            reasons.append("windows_checkpoint_mismatch")
        if windows.get("promotion_ceremony_sha256") != ceremony.get(
            "ceremony_sha256"
        ):
            reasons.append("windows_ceremony_mismatch")

    if device is not None and ceremony is not None:
        if device.get("candidate_checkpoint_sha256") != ceremony.get(
            "candidate_checkpoint_sha256"
        ):
            reasons.append("android_device_checkpoint_mismatch")
        if device.get("promotion_ceremony_sha256") != ceremony.get(
            "ceremony_sha256"
        ):
            reasons.append("android_device_ceremony_mismatch")
        if str(device.get("product_version")) != str(expected_product_version):
            reasons.append("android_device_product_version_mismatch")

    if performance is not None and device is not None:
        if performance.get("apk_sha256") != device.get("apk_sha256"):
            reasons.append("android_performance_apk_mismatch")
        if performance.get("candidate_checkpoint_sha256") != device.get(
            "candidate_checkpoint_sha256"
        ):
            reasons.append("android_performance_checkpoint_mismatch")
        if performance.get("device_fingerprint_sha256") != device.get(
            "device_fingerprint_sha256"
        ):
            reasons.append("android_performance_device_mismatch")

    reasons = sorted(set(reasons))
    receipt: dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "authority": CLOSURE_AUTHORITY,
        "status": READY_STATUS if not reasons else BLOCKED_STATUS,
        "reasons": reasons,
        "expected_product_version": str(expected_product_version),
        "candidate_checkpoint_sha256": (
            ceremony.get("candidate_checkpoint_sha256")
            if ceremony is not None
            else None
        ),
        "evidence": evidence,
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_chat_text": False,
            "contains_device_fingerprint": False,
            "contains_local_paths": False,
        },
    }
    receipt["closure_sha256"] = payload_digest(receipt)
    return receipt


def verify_v1_closure_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != CLOSURE_SCHEMA:
        raise ValueError("unsupported v1 closure receipt schema")
    if receipt.get("authority") != CLOSURE_AUTHORITY:
        raise ValueError("v1 closure receipt authority mismatch")
    _verify_self_digest(
        receipt,
        digest_field="closure_sha256",
        label="v1 closure receipt",
    )
    if receipt.get("status") not in {READY_STATUS, BLOCKED_STATUS}:
        raise ValueError("v1 closure status invalid")
    reasons = receipt.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("v1 closure reasons invalid")
    if receipt.get("status") == READY_STATUS and reasons:
        raise ValueError("READY v1 closure receipt has blocking reasons")
    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict) or any(bool(privacy.get(key, True)) for key in (
        "contains_raw_prompt_target",
        "contains_chat_text",
        "contains_device_fingerprint",
        "contains_local_paths",
    )):
        raise ValueError("v1 closure receipt privacy invalid")
    return receipt

CI_CLOSURE_SCHEMA = "NOLANE-V060-CI-SOFTWARE-CLOSURE-V1"
CI_CLOSURE_AUTHORITY = "CI_SOFTWARE_RELEASE_READINESS_NO_HARDWARE_CERTIFICATION"
CI_READY_STATUS = "READY_FOR_V1_0_CI_VERIFIED"
CI_REQUIRED_WORKFLOWS = (
    "Product Client Court",
    "Living Runtime CI",
    "Neural Shadow CI",
    "Platform Crash Court",
)
CI_REQUIRED_PRODUCT_JOBS = (
    "Product runtime court",
    "Android APK court",
    "NUI browser court",
    "Android native kernel",
    "Windows native host",
    "Android x86_64 emulator APK",
    "Android emulator local-chat restart court",
)


def _git_sha(value: Any, *, field: str = "commit_sha") -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{field} must be a 40-character Git SHA")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a 40-character Git SHA") from exc
    return value.lower()


def build_ci_v1_closure_receipt(
    *,
    repository: str,
    branch: str,
    commit_sha: str,
    product_version: str,
    workflows: dict[str, str],
    product_jobs: dict[str, str],
) -> dict[str, Any]:
    reasons: list[str] = []
    normalized_sha = _git_sha(commit_sha)

    if branch != "main":
        reasons.append("not_main_branch")

    for name in CI_REQUIRED_WORKFLOWS:
        conclusion = str(workflows.get(name, "missing"))
        if conclusion != "success":
            reasons.append(
                f"workflow_not_success:{name}:{conclusion}"
            )

    for name in CI_REQUIRED_PRODUCT_JOBS:
        conclusion = str(product_jobs.get(name, "missing"))
        if conclusion != "success":
            reasons.append(
                f"product_job_not_success:{name}:{conclusion}"
            )

    reasons = sorted(set(reasons))
    receipt: dict[str, Any] = {
        "schema": CI_CLOSURE_SCHEMA,
        "authority": CI_CLOSURE_AUTHORITY,
        "status": CI_READY_STATUS if not reasons else BLOCKED_STATUS,
        "reasons": reasons,
        "repository": str(repository),
        "branch": str(branch),
        "commit_sha": normalized_sha,
        "product_version": str(product_version),
        "workflows": {
            name: str(workflows.get(name, "missing"))
            for name in CI_REQUIRED_WORKFLOWS
        },
        "product_jobs": {
            name: str(product_jobs.get(name, "missing"))
            for name in CI_REQUIRED_PRODUCT_JOBS
        },
        "verified_scope": [
            "python_runtime_contracts",
            "neural_shadow_contracts",
            "platform_crash_recovery",
            "windows_native_packaging",
            "android_arm64_packaging",
            "android_x86_64_packaging",
            "android_emulator_native_boot",
            "android_force_stop_restart_continuity",
            "localmobile_native_inference",
            "persistent_identity_state_history",
            "mobile_lifecycle_parity",
            "authority_bound_release_mechanics",
        ],
        "excluded_claims": [
            "physical_device_certification",
            "field_battery_runtime",
            "field_thermal_behavior",
            "field_radio_or_oem_compatibility",
            "real_world_performance_distribution",
        ],
        "privacy": {
            "contains_chat_text": False,
            "contains_auth_token": False,
            "contains_device_fingerprint": False,
            "contains_local_paths": False,
        },
    }
    receipt["closure_sha256"] = payload_digest(receipt)
    return verify_ci_v1_closure_receipt(receipt)


def verify_ci_v1_closure_receipt(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != CI_CLOSURE_SCHEMA:
        raise ValueError("unsupported CI v1 closure receipt schema")
    if receipt.get("authority") != CI_CLOSURE_AUTHORITY:
        raise ValueError("CI v1 closure authority mismatch")
    _verify_self_digest(
        receipt,
        digest_field="closure_sha256",
        label="CI v1 closure receipt",
    )
    if receipt.get("status") not in {CI_READY_STATUS, BLOCKED_STATUS}:
        raise ValueError("CI v1 closure status invalid")
    _git_sha(receipt.get("commit_sha"))
    reasons = receipt.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("CI v1 closure reasons invalid")
    if receipt.get("status") == CI_READY_STATUS and reasons:
        raise ValueError("READY CI closure has blocking reasons")
    if receipt.get("branch") != "main" and receipt.get("status") == CI_READY_STATUS:
        raise ValueError("CI v1 closure cannot be READY off main")

    workflows = receipt.get("workflows")
    jobs = receipt.get("product_jobs")
    if not isinstance(workflows, dict) or not isinstance(jobs, dict):
        raise ValueError("CI v1 closure evidence maps missing")

    if receipt.get("status") == CI_READY_STATUS:
        if any(workflows.get(name) != "success" for name in CI_REQUIRED_WORKFLOWS):
            raise ValueError("READY CI closure has failed workflow")
        if any(jobs.get(name) != "success" for name in CI_REQUIRED_PRODUCT_JOBS):
            raise ValueError("READY CI closure has failed product job")

    excluded = receipt.get("excluded_claims")
    if not isinstance(excluded, list):
        raise ValueError("CI v1 closure excluded_claims missing")
    required_exclusions = {
        "physical_device_certification",
        "field_battery_runtime",
        "field_thermal_behavior",
        "field_radio_or_oem_compatibility",
        "real_world_performance_distribution",
    }
    if not required_exclusions.issubset(set(excluded)):
        raise ValueError("CI v1 closure overclaims hardware evidence")

    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict) or any(
        bool(privacy.get(key, True))
        for key in (
            "contains_chat_text",
            "contains_auth_token",
            "contains_device_fingerprint",
            "contains_local_paths",
        )
    ):
        raise ValueError("CI v1 closure privacy invalid")
    return receipt

