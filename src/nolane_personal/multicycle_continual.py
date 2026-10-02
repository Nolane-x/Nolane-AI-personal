from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .continual_learning_court import verify_continual_learning_digest
from .long_horizon_retention import (
    verify_long_horizon_retention_digest,
)
from .store import payload_digest


SCHEMA = "NOLANE-L32-MULTICYCLE-CONTINUAL-CHAIN-V1"
L31_RUN_SCHEMA = "NOLANE-L31-CONTINUAL-FACTORIZED-UPDATE-RUN-V1"
L31_TRAINING_SCHEMA = "NOLANE-L31-FACTORIZED-CONTINUAL-UPDATE-V1"
L17_ARTIFACT_SCHEMA = "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1"


@dataclass(slots=True)
class MultiCycleContinualPolicy:
    min_cycles: int = 2
    require_unique_adaptation_protocols: bool = True

    def validate(self) -> None:
        if self.min_cycles < 2:
            raise ValueError("min_cycles must be >=2")


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def _verify_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    if lineage.get("schema") != "NOLANE-L31-CONTINUAL-UPDATE-LINEAGE-V1":
        raise ValueError("unsupported L31 lineage schema")
    supplied = lineage.get("lineage_sha256")
    body = dict(lineage)
    body.pop("lineage_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("L31 lineage digest mismatch")
    for key in (
        "parent_factorized_checkpoint_sha256",
        "retention_dataset_sha256",
        "retention_protocol_sha256",
        "retention_quality_court_sha256",
        "adaptation_dataset_sha256",
        "adaptation_protocol_sha256",
        "adaptation_quality_court_sha256",
        "lineage_sha256",
    ):
        _validate_sha256(lineage.get(key), name=key)
    return lineage


def verify_l31_run_receipt(run: dict[str, Any]) -> dict[str, Any]:
    if run.get("schema") != L31_RUN_SCHEMA:
        raise ValueError("unsupported L31 run receipt schema")
    if run.get("authority") != "CONTINUAL_UPDATE_EVIDENCE_ONLY_UNPROMOTED":
        raise ValueError("L31 run authority mismatch")

    parent_sha = _validate_sha256(
        run.get("parent_factorized_checkpoint_sha256"),
        name="parent_factorized_checkpoint_sha256",
    )
    training = run.get("training")
    artifact = run.get("artifact")
    if not isinstance(training, dict) or not isinstance(artifact, dict):
        raise ValueError("L31 run training/artifact evidence missing")

    if training.get("schema") != L31_TRAINING_SCHEMA:
        raise ValueError("unsupported L31 training receipt schema")
    if training.get("authority") != "CONTINUAL_FACTORIZED_UPDATE_CANDIDATE_ONLY":
        raise ValueError("L31 training authority mismatch")
    if not training.get("candidate_boundary_changed"):
        raise ValueError("L31 candidate boundary did not change")
    if not training.get("reference_boundary_unchanged"):
        raise ValueError("L31 reference boundary changed")
    if not training.get("candidate_cortex_unchanged"):
        raise ValueError("L31 candidate cortex changed")
    if not training.get("reference_cortex_unchanged"):
        raise ValueError("L31 reference cortex changed")
    if int(training.get("candidate_cortex_gradients_seen", -1)) != 0:
        raise ValueError("L31 candidate cortex received gradients")

    continual = training.get("continual_learning")
    if not isinstance(continual, dict):
        raise ValueError("L31 continual-learning court missing")
    verify_continual_learning_digest(continual)
    if continual.get("status") != "PASS":
        raise ValueError("L31 continual-learning court did not pass")

    lineage = training.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("L31 lineage missing")
    _verify_lineage(lineage)
    if lineage["parent_factorized_checkpoint_sha256"] != parent_sha:
        raise ValueError("L31 lineage parent checkpoint mismatch")

    if artifact.get("schema") != L17_ARTIFACT_SCHEMA:
        raise ValueError("L31 output artifact schema mismatch")
    if artifact.get("authority") != "FACTORIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("L31 output artifact authority mismatch")
    artifact_sha = _validate_sha256(
        artifact.get("checkpoint_sha256"),
        name="artifact checkpoint_sha256",
    )
    boundary_after = _validate_sha256(
        training.get("candidate_boundary_digest_after"),
        name="candidate_boundary_digest_after",
    )
    boundary_before = _validate_sha256(
        training.get("candidate_boundary_digest_before"),
        name="candidate_boundary_digest_before",
    )
    cortex_after = _validate_sha256(
        training.get("candidate_cortex_digest_after"),
        name="candidate_cortex_digest_after",
    )
    if artifact.get("boundary_state_digest") != boundary_after:
        raise ValueError("L31 saved boundary digest mismatch")
    if artifact.get("cortex_state_digest") != cortex_after:
        raise ValueError("L31 saved cortex digest mismatch")
    if artifact.get("dataset_fingerprint") != lineage["lineage_sha256"]:
        raise ValueError("L31 artifact lineage fingerprint mismatch")
    if artifact.get("training_receipt") != training:
        raise ValueError("L31 artifact training receipt mismatch")
    if continual.get("pre_update_checkpoint_sha256") != boundary_before:
        raise ValueError("L31 L30 pre-update digest mismatch")
    if continual.get("post_update_checkpoint_sha256") != boundary_after:
        raise ValueError("L31 L30 post-update digest mismatch")

    return run


def assess_multicycle_continual_chain(
    cycles: list[dict[str, Any]],
    *,
    long_horizon_retention: dict[str, Any],
    policy: MultiCycleContinualPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or MultiCycleContinualPolicy()
    policy.validate()
    if len(cycles) < policy.min_cycles:
        raise ValueError(
            f"need at least {policy.min_cycles} L31 update cycles"
        )

    verified = [verify_l31_run_receipt(cycle) for cycle in cycles]
    verify_long_horizon_retention_digest(long_horizon_retention)

    reasons: list[str] = []
    summaries: list[dict[str, Any]] = []
    adaptation_protocols: list[str] = []
    positive_mean_retention_regression = 0.0
    worst_cycle_retention_regression = float("-inf")
    previous_artifact_sha: str | None = None
    previous_boundary_after: str | None = None

    for index, cycle in enumerate(verified):
        training = cycle["training"]
        lineage = training["lineage"]
        artifact = cycle["artifact"]
        continual = training["continual_learning"]
        parent_sha = cycle["parent_factorized_checkpoint_sha256"]
        artifact_sha = artifact["checkpoint_sha256"]
        boundary_before = training["candidate_boundary_digest_before"]
        boundary_after = training["candidate_boundary_digest_after"]

        if previous_artifact_sha is not None and parent_sha != previous_artifact_sha:
            reasons.append("artifact_chain_discontinuity")
        if (
            previous_boundary_after is not None
            and boundary_before != previous_boundary_after
        ):
            reasons.append("boundary_state_chain_discontinuity")

        adaptation_protocol = lineage["adaptation_protocol_sha256"]
        adaptation_protocols.append(adaptation_protocol)

        retention_summary = continual["retention"]["summary"]
        adaptation_summary = continual["adaptation"]["summary"]
        mean_retention = float(retention_summary["mean_group_regression"])
        worst_retention = float(retention_summary["worst_group_regression"])
        positive_mean_retention_regression += max(0.0, mean_retention)
        worst_cycle_retention_regression = max(
            worst_cycle_retention_regression,
            worst_retention,
        )

        summaries.append({
            "cycle": index,
            "parent_factorized_checkpoint_sha256": parent_sha,
            "artifact_checkpoint_sha256": artifact_sha,
            "boundary_digest_before": boundary_before,
            "boundary_digest_after": boundary_after,
            "retention_mean_group_regression": round(mean_retention, 8),
            "retention_worst_group_regression": round(worst_retention, 8),
            "adaptation_mean_group_gain": round(
                float(adaptation_summary["mean_group_gain"]),
                8,
            ),
            "continual_learning_court_sha256": continual["court_sha256"],
            "lineage_sha256": lineage["lineage_sha256"],
        })

        previous_artifact_sha = artifact_sha
        previous_boundary_after = boundary_after

    if (
        policy.require_unique_adaptation_protocols
        and len(set(adaptation_protocols)) != len(adaptation_protocols)
    ):
        reasons.append("adaptation_protocol_reused_across_cycles")

    first_parent = cycles[0]["parent_factorized_checkpoint_sha256"]
    final_artifact = cycles[-1]["artifact"]["checkpoint_sha256"]
    if long_horizon_retention.get("initial_checkpoint_sha256") != first_parent:
        reasons.append("long_horizon_initial_checkpoint_mismatch")
    if long_horizon_retention.get("final_checkpoint_sha256") != final_artifact:
        reasons.append("long_horizon_final_checkpoint_mismatch")
    if long_horizon_retention.get("status") != "PASS":
        reasons.append("long_horizon_fixed_panel_failed")

    receipt = {
        "schema": SCHEMA,
        "authority": "MULTICYCLE_CONTINUAL_CHAIN_ONLY_NO_PRODUCTION_AUTHORITY",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "policy": asdict(policy),
        "cycles": len(cycles),
        "first_parent_checkpoint_sha256": first_parent,
        "final_artifact_checkpoint_sha256": final_artifact,
        "cycle_summaries": summaries,
        "summary": {
            "positive_mean_retention_regression_sum": round(
                positive_mean_retention_regression,
                8,
            ),
            "worst_cycle_retention_group_regression": round(
                worst_cycle_retention_regression,
                8,
            ),
            "unique_adaptation_protocols": len(set(adaptation_protocols)),
        },
        "long_horizon_retention_court_sha256": (
            long_horizon_retention["court_sha256"]
        ),
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_source_group_hash_values": False,
        },
    }
    receipt["chain_sha256"] = payload_digest(receipt)
    return receipt


def verify_multicycle_chain_digest(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported multicycle continual chain schema")
    supplied = receipt.get("chain_sha256")
    body = dict(receipt)
    body.pop("chain_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("multicycle continual chain digest mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("multicycle continual chain status invalid")
    return receipt
