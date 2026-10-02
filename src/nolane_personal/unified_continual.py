from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .continual_cortex_update import verify_continual_cortex_update_receipt
from .long_horizon_retention import verify_long_horizon_retention_digest
from .multicycle_continual import verify_l31_run_receipt
from .store import payload_digest


SCHEMA = "NOLANE-L39-UNIFIED-CONTINUAL-MODEL-CHAIN-V1"
L38_RUN_SCHEMA = "NOLANE-L38-RECURRENT-CORTEX-UPDATE-RUN-V1"
L38_LINEAGE_SCHEMA = "NOLANE-L38-CORTEX-UPDATE-LINEAGE-V1"
L17_ARTIFACT_SCHEMA = "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1"


@dataclass(slots=True)
class UnifiedContinualPolicy:
    min_cycles: int = 2
    min_cortex_cycles: int = 1
    require_unique_adaptation_protocols: bool = True

    def validate(self) -> None:
        if self.min_cycles < 2:
            raise ValueError("min_cycles must be >=2")
        if self.min_cortex_cycles < 1:
            raise ValueError("min_cortex_cycles must be >=1")


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def model_state_sha256(
    *,
    boundary_state_digest: str,
    cortex_state_digest: str,
) -> str:
    boundary = _validate_sha256(
        boundary_state_digest,
        name="boundary_state_digest",
    )
    cortex = _validate_sha256(
        cortex_state_digest,
        name="cortex_state_digest",
    )
    return payload_digest(
        {
            "boundary_state_digest": boundary,
            "cortex_state_digest": cortex,
        }
    )


def _verify_l38_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    if lineage.get("schema") != L38_LINEAGE_SCHEMA:
        raise ValueError("unsupported L38 lineage schema")
    supplied = lineage.get("lineage_sha256")
    body = dict(lineage)
    body.pop("lineage_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("L38 lineage digest mismatch")
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


def verify_l38_run_receipt(run: dict[str, Any]) -> dict[str, Any]:
    if run.get("schema") != L38_RUN_SCHEMA:
        raise ValueError("unsupported L38 run receipt schema")
    if run.get("authority") != "RECURRENT_CORTEX_UPDATE_EVIDENCE_ONLY_UNPROMOTED":
        raise ValueError("L38 run authority mismatch")

    parent_sha = _validate_sha256(
        run.get("parent_factorized_checkpoint_sha256"),
        name="parent_factorized_checkpoint_sha256",
    )
    training = run.get("training")
    artifact = run.get("artifact")
    if not isinstance(training, dict) or not isinstance(artifact, dict):
        raise ValueError("L38 run training/artifact evidence missing")

    lineage = training.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("L38 lineage missing")
    _verify_l38_lineage(lineage)
    if lineage["parent_factorized_checkpoint_sha256"] != parent_sha:
        raise ValueError("L38 lineage parent checkpoint mismatch")

    core_training = dict(training)
    core_training.pop("lineage", None)
    verify_continual_cortex_update_receipt(core_training)
    continual = core_training["continual_learning"]
    if continual.get("status") != "PASS":
        raise ValueError("L38 continual-learning court did not pass")

    boundary_before = _validate_sha256(
        core_training.get("candidate_boundary_digest_before"),
        name="candidate_boundary_digest_before",
    )
    boundary_after = _validate_sha256(
        core_training.get("candidate_boundary_digest_after"),
        name="candidate_boundary_digest_after",
    )
    cortex_before = _validate_sha256(
        core_training.get("candidate_cortex_digest_before"),
        name="candidate_cortex_digest_before",
    )
    cortex_after = _validate_sha256(
        core_training.get("candidate_cortex_digest_after"),
        name="candidate_cortex_digest_after",
    )
    model_before = model_state_sha256(
        boundary_state_digest=boundary_before,
        cortex_state_digest=cortex_before,
    )
    model_after = model_state_sha256(
        boundary_state_digest=boundary_after,
        cortex_state_digest=cortex_after,
    )
    if core_training.get("candidate_model_state_sha256_before") != model_before:
        raise ValueError("L38 model-state before digest mismatch")
    if core_training.get("candidate_model_state_sha256_after") != model_after:
        raise ValueError("L38 model-state after digest mismatch")
    if continual.get("pre_update_checkpoint_sha256") != model_before:
        raise ValueError("L38 L30 pre-update model-state mismatch")
    if continual.get("post_update_checkpoint_sha256") != model_after:
        raise ValueError("L38 L30 post-update model-state mismatch")

    if artifact.get("schema") != L17_ARTIFACT_SCHEMA:
        raise ValueError("L38 output artifact schema mismatch")
    if artifact.get("authority") != "FACTORIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("L38 output artifact authority mismatch")
    _validate_sha256(
        artifact.get("checkpoint_sha256"),
        name="artifact checkpoint_sha256",
    )
    if artifact.get("boundary_state_digest") != boundary_after:
        raise ValueError("L38 saved boundary digest mismatch")
    if artifact.get("cortex_state_digest") != cortex_after:
        raise ValueError("L38 saved cortex digest mismatch")
    if artifact.get("dataset_fingerprint") != lineage["lineage_sha256"]:
        raise ValueError("L38 artifact lineage fingerprint mismatch")
    if artifact.get("training_receipt") != training:
        raise ValueError("L38 artifact training receipt mismatch")
    return run


def normalize_continual_cycle(run: dict[str, Any]) -> dict[str, Any]:
    schema = run.get("schema")
    if schema == "NOLANE-L31-CONTINUAL-FACTORIZED-UPDATE-RUN-V1":
        verify_l31_run_receipt(run)
        training = run["training"]
        lineage = training["lineage"]
        boundary_before = training["candidate_boundary_digest_before"]
        boundary_after = training["candidate_boundary_digest_after"]
        cortex_before = training["candidate_cortex_digest_before"]
        cortex_after = training["candidate_cortex_digest_after"]
        cycle_type = "BOUNDARY_UPDATE"
    elif schema == L38_RUN_SCHEMA:
        verify_l38_run_receipt(run)
        training = run["training"]
        lineage = training["lineage"]
        boundary_before = training["candidate_boundary_digest_before"]
        boundary_after = training["candidate_boundary_digest_after"]
        cortex_before = training["candidate_cortex_digest_before"]
        cortex_after = training["candidate_cortex_digest_after"]
        cycle_type = "CORTEX_UPDATE"
    else:
        raise ValueError("unsupported continual cycle schema")

    artifact = run["artifact"]
    continual = training["continual_learning"]
    return {
        "cycle_type": cycle_type,
        "run_schema": schema,
        "parent_artifact_sha256": run["parent_factorized_checkpoint_sha256"],
        "artifact_sha256": artifact["checkpoint_sha256"],
        "boundary_before": boundary_before,
        "boundary_after": boundary_after,
        "cortex_before": cortex_before,
        "cortex_after": cortex_after,
        "model_state_before": model_state_sha256(
            boundary_state_digest=boundary_before,
            cortex_state_digest=cortex_before,
        ),
        "model_state_after": model_state_sha256(
            boundary_state_digest=boundary_after,
            cortex_state_digest=cortex_after,
        ),
        "adaptation_protocol_sha256": lineage["adaptation_protocol_sha256"],
        "lineage_sha256": lineage["lineage_sha256"],
        "continual_learning_court_sha256": continual["court_sha256"],
        "retention_mean_group_regression": float(
            continual["retention"]["summary"]["mean_group_regression"]
        ),
        "retention_worst_group_regression": float(
            continual["retention"]["summary"]["worst_group_regression"]
        ),
        "adaptation_mean_group_gain": float(
            continual["adaptation"]["summary"]["mean_group_gain"]
        ),
    }


def assess_unified_continual_chain(
    cycles: list[dict[str, Any]],
    *,
    long_horizon_retention: dict[str, Any],
    policy: UnifiedContinualPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or UnifiedContinualPolicy()
    policy.validate()
    if len(cycles) < policy.min_cycles:
        raise ValueError(
            f"need at least {policy.min_cycles} continual update cycles"
        )
    verify_long_horizon_retention_digest(long_horizon_retention)

    normalized = [normalize_continual_cycle(cycle) for cycle in cycles]
    reasons: list[str] = []
    adaptation_protocols: list[str] = []
    cortex_cycles = 0
    boundary_cycles = 0
    positive_mean_retention_regression = 0.0
    worst_cycle_retention_regression = float("-inf")

    summaries: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for index, cycle in enumerate(normalized):
        if cycle["cycle_type"] == "CORTEX_UPDATE":
            cortex_cycles += 1
        elif cycle["cycle_type"] == "BOUNDARY_UPDATE":
            boundary_cycles += 1

        if previous is not None:
            if cycle["parent_artifact_sha256"] != previous["artifact_sha256"]:
                reasons.append("artifact_chain_discontinuity")
            if cycle["boundary_before"] != previous["boundary_after"]:
                reasons.append("boundary_state_chain_discontinuity")
            if cycle["cortex_before"] != previous["cortex_after"]:
                reasons.append("cortex_state_chain_discontinuity")
            if cycle["model_state_before"] != previous["model_state_after"]:
                reasons.append("model_state_chain_discontinuity")

        adaptation_protocols.append(cycle["adaptation_protocol_sha256"])
        positive_mean_retention_regression += max(
            0.0,
            cycle["retention_mean_group_regression"],
        )
        worst_cycle_retention_regression = max(
            worst_cycle_retention_regression,
            cycle["retention_worst_group_regression"],
        )

        summaries.append(
            {
                "cycle": index,
                "cycle_type": cycle["cycle_type"],
                "run_schema": cycle["run_schema"],
                "parent_artifact_sha256": cycle["parent_artifact_sha256"],
                "artifact_sha256": cycle["artifact_sha256"],
                "boundary_before": cycle["boundary_before"],
                "boundary_after": cycle["boundary_after"],
                "cortex_before": cycle["cortex_before"],
                "cortex_after": cycle["cortex_after"],
                "model_state_before": cycle["model_state_before"],
                "model_state_after": cycle["model_state_after"],
                "retention_mean_group_regression": round(
                    cycle["retention_mean_group_regression"],
                    8,
                ),
                "retention_worst_group_regression": round(
                    cycle["retention_worst_group_regression"],
                    8,
                ),
                "adaptation_mean_group_gain": round(
                    cycle["adaptation_mean_group_gain"],
                    8,
                ),
                "continual_learning_court_sha256": (
                    cycle["continual_learning_court_sha256"]
                ),
                "lineage_sha256": cycle["lineage_sha256"],
            }
        )
        previous = cycle

    if cortex_cycles < policy.min_cortex_cycles:
        reasons.append("insufficient_recurrent_cortex_cycles")
    if (
        policy.require_unique_adaptation_protocols
        and len(set(adaptation_protocols)) != len(adaptation_protocols)
    ):
        reasons.append("adaptation_protocol_reused_across_cycles")

    first_parent = normalized[0]["parent_artifact_sha256"]
    final_artifact = normalized[-1]["artifact_sha256"]
    if long_horizon_retention.get("initial_checkpoint_sha256") != first_parent:
        reasons.append("long_horizon_initial_checkpoint_mismatch")
    if long_horizon_retention.get("final_checkpoint_sha256") != final_artifact:
        reasons.append("long_horizon_final_checkpoint_mismatch")
    if long_horizon_retention.get("status") != "PASS":
        reasons.append("long_horizon_fixed_panel_failed")

    receipt = {
        "schema": SCHEMA,
        "authority": "UNIFIED_CONTINUAL_MODEL_CHAIN_NO_PRODUCTION_AUTHORITY",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "policy": asdict(policy),
        "cycles": len(normalized),
        "boundary_cycles": boundary_cycles,
        "cortex_cycles": cortex_cycles,
        "first_parent_checkpoint_sha256": first_parent,
        "final_artifact_checkpoint_sha256": final_artifact,
        "initial_model_state_sha256": normalized[0]["model_state_before"],
        "final_model_state_sha256": normalized[-1]["model_state_after"],
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


def verify_unified_continual_chain_digest(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported unified continual chain schema")
    supplied = receipt.get("chain_sha256")
    body = dict(receipt)
    body.pop("chain_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("unified continual chain digest mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("unified continual chain status invalid")
    return receipt
