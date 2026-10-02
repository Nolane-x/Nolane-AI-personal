from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .heldout_group_robustness import (
    HeldoutGroupRobustnessPolicy,
    assess_group_robustness,
    verify_group_robustness_digest,
)
from .store import payload_digest


SCHEMA = "NOLANE-L32-LONG-HORIZON-RETENTION-V1"
FLOAT_EPSILON = 1e-12


@dataclass(slots=True)
class LongHorizonRetentionPolicy:
    min_groups: int = 2
    max_overall_regression: float = 0.01
    max_worst_group_regression: float = 0.03

    def validate(self) -> None:
        if self.min_groups < 2:
            raise ValueError("min_groups must be >=2")
        if self.max_overall_regression < 0:
            raise ValueError("max_overall_regression must be non-negative")
        if self.max_worst_group_regression < 0:
            raise ValueError(
                "max_worst_group_regression must be non-negative"
            )


def _validate_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def assess_long_horizon_retention(
    protocol: dict[str, Any],
    *,
    initial_checkpoint_sha256: str,
    final_checkpoint_sha256: str,
    initial_values: list[float],
    final_values: list[float],
    policy: LongHorizonRetentionPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or LongHorizonRetentionPolicy()
    policy.validate()
    initial_checkpoint_sha256 = _validate_sha256(
        initial_checkpoint_sha256,
        name="initial_checkpoint_sha256",
    )
    final_checkpoint_sha256 = _validate_sha256(
        final_checkpoint_sha256,
        name="final_checkpoint_sha256",
    )

    reasons: list[str] = []
    if initial_checkpoint_sha256 == final_checkpoint_sha256:
        reasons.append("checkpoint_did_not_change")

    group = assess_group_robustness(
        protocol,
        split="test",
        reference_values=initial_values,
        candidate_values=final_values,
        policy=HeldoutGroupRobustnessPolicy(
            min_groups=policy.min_groups,
            max_worst_group_regression=(
                policy.max_worst_group_regression
            ),
        ),
    )
    if group["status"] != "PASS":
        reasons.append("fixed_panel_group_retention_failed")

    overall = group["summary"]["overall_regression"]
    if overall is None:
        reasons.append("fixed_panel_overall_regression_missing")
    elif (
        float(overall)
        > policy.max_overall_regression + FLOAT_EPSILON
    ):
        reasons.append("fixed_panel_overall_retention_failed")

    receipt = {
        "schema": SCHEMA,
        "authority": (
            "FIXED_PANEL_LONG_HORIZON_RETENTION_ONLY_NO_PROMOTION_AUTHORITY"
        ),
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "protocol_sha256": str(protocol.get("protocol_sha256")),
        "policy": asdict(policy),
        "group_robustness": group,
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_source_group_hash_values": False,
        },
    }
    receipt["court_sha256"] = payload_digest(receipt)
    return receipt


def verify_long_horizon_retention_digest(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported long-horizon retention schema")
    supplied = receipt.get("court_sha256")
    body = dict(receipt)
    body.pop("court_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("long-horizon retention receipt digest mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("long-horizon retention status invalid")
    group = receipt.get("group_robustness")
    if not isinstance(group, dict):
        raise ValueError("long-horizon group court missing")
    verify_group_robustness_digest(group)
    return receipt
