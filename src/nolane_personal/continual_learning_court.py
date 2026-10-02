from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any

from .store import payload_digest


SCHEMA = "NOLANE-L30-LONG-HORIZON-CONTINUAL-LEARNING-V1"\nFLOAT_EPSILON = 1e-12


@dataclass(slots=True)
class ContinualLearningPolicy:
    min_retention_groups: int = 2
    min_adaptation_groups: int = 2
    max_worst_retention_regression: float = 0.03
    max_mean_retention_regression: float = 0.01
    min_mean_adaptation_gain: float = 0.0
    max_worst_adaptation_regression: float = 0.03

    def validate(self) -> None:
        if self.min_retention_groups < 2:
            raise ValueError("min_retention_groups must be >=2")
        if self.min_adaptation_groups < 1:
            raise ValueError("min_adaptation_groups must be >=1")
        for name in (
            "max_worst_retention_regression",
            "max_mean_retention_regression",
            "max_worst_adaptation_regression",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(float(self.min_mean_adaptation_gain)):
            raise ValueError("min_mean_adaptation_gain must be finite")


def _validate_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def _validate_metric_rows(
    *,
    group_sha256: list[str],
    before_values: list[float],
    after_values: list[float],
    label: str,
) -> None:
    if not (
        len(group_sha256) == len(before_values) == len(after_values)
    ):
        raise ValueError(f"{label} metric/group lengths do not match")
    if not group_sha256:
        raise ValueError(f"{label} evidence is empty")
    for value in group_sha256:
        _validate_digest(value, name=f"{label} source group")
    if any(
        not math.isfinite(float(value))
        for value in before_values + after_values
    ):
        raise ValueError(f"{label} metrics must be finite")


def _group_regressions(
    group_sha256: list[str],
    before_values: list[float],
    after_values: list[float],
) -> tuple[list[dict[str, Any]], list[float]]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index, group in enumerate(group_sha256):
        if group not in grouped:
            grouped[group] = []
            order.append(group)
        grouped[group].append(index)

    rows: list[dict[str, Any]] = []
    regressions: list[float] = []
    for ordinal, group in enumerate(order):
        indices = grouped[group]
        before = sum(float(before_values[i]) for i in indices) / len(indices)
        after = sum(float(after_values[i]) for i in indices) / len(indices)
        regression = after - before
        regressions.append(regression)
        rows.append(
            {
                "group_alias": f"group_{ordinal:03d}",
                "examples": len(indices),
                "local_indices": list(indices),
                "before_mean": round(before, 8),
                "after_mean": round(after, 8),
                "regression": round(regression, 8),
                "gain": round(-regression, 8),
            }
        )
    return rows, regressions


def assess_continual_learning(
    *,
    pre_update_checkpoint_sha256: str,
    post_update_checkpoint_sha256: str,
    retention_group_sha256: list[str],
    retention_before_values: list[float],
    retention_after_values: list[float],
    adaptation_group_sha256: list[str],
    adaptation_before_values: list[float],
    adaptation_after_values: list[float],
    policy: ContinualLearningPolicy | None = None,
) -> dict[str, Any]:
    """
    Assess stability/plasticity for a sequential model update.

    All metrics are assumed to be lower-is-better and must be measured on the
    same examples before and after the update. Retention rows are old evidence;
    adaptation rows are newly introduced evidence. Raw group hashes are never
    copied into the receipt.
    """
    policy = policy or ContinualLearningPolicy()
    policy.validate()

    pre = _validate_digest(
        pre_update_checkpoint_sha256,
        name="pre_update_checkpoint_sha256",
    )
    post = _validate_digest(
        post_update_checkpoint_sha256,
        name="post_update_checkpoint_sha256",
    )
    _validate_metric_rows(
        group_sha256=retention_group_sha256,
        before_values=retention_before_values,
        after_values=retention_after_values,
        label="retention",
    )
    _validate_metric_rows(
        group_sha256=adaptation_group_sha256,
        before_values=adaptation_before_values,
        after_values=adaptation_after_values,
        label="adaptation",
    )

    reasons: list[str] = []
    retention_groups = list(dict.fromkeys(retention_group_sha256))
    adaptation_groups = list(dict.fromkeys(adaptation_group_sha256))

    if pre == post:
        reasons.append("checkpoint_did_not_change")
    if len(retention_groups) < policy.min_retention_groups:
        reasons.append("insufficient_retention_groups")
    if len(adaptation_groups) < policy.min_adaptation_groups:
        reasons.append("insufficient_adaptation_groups")
    if set(retention_groups) & set(adaptation_groups):
        reasons.append("retention_adaptation_group_overlap")

    retention_rows, retention_regressions = _group_regressions(
        retention_group_sha256,
        retention_before_values,
        retention_after_values,
    )
    adaptation_rows, adaptation_regressions = _group_regressions(
        adaptation_group_sha256,
        adaptation_before_values,
        adaptation_after_values,
    )

    retention_mean = sum(retention_regressions) / len(retention_regressions)
    retention_worst = max(retention_regressions)
    retention_best = min(retention_regressions)
    retention_dispersion = (
        pstdev(retention_regressions)
        if len(retention_regressions) >= 2
        else 0.0
    )

    adaptation_mean_regression = (
        sum(adaptation_regressions) / len(adaptation_regressions)
    )
    adaptation_worst_regression = max(adaptation_regressions)
    adaptation_best_regression = min(adaptation_regressions)
    adaptation_mean_gain = -adaptation_mean_regression
    adaptation_dispersion = (
        pstdev(adaptation_regressions)
        if len(adaptation_regressions) >= 2
        else 0.0
    )

    if retention_worst > policy.max_worst_retention_regression + FLOAT_EPSILON:
        reasons.append("worst_retention_forgetting_failed")
    if retention_mean > policy.max_mean_retention_regression + FLOAT_EPSILON:
        reasons.append("mean_retention_forgetting_failed")
    if adaptation_mean_gain < policy.min_mean_adaptation_gain - FLOAT_EPSILON:
        reasons.append("adaptation_gain_failed")
    if adaptation_worst_regression > policy.max_worst_adaptation_regression + FLOAT_EPSILON:
        reasons.append("worst_adaptation_group_failed")

    retention_before = (
        sum(float(v) for v in retention_before_values)
        / len(retention_before_values)
    )
    retention_after = (
        sum(float(v) for v in retention_after_values)
        / len(retention_after_values)
    )
    adaptation_before = (
        sum(float(v) for v in adaptation_before_values)
        / len(adaptation_before_values)
    )
    adaptation_after = (
        sum(float(v) for v in adaptation_after_values)
        / len(adaptation_after_values)
    )

    receipt = {
        "schema": SCHEMA,
        "authority": "CONTINUAL_LEARNING_STABILITY_PLASTICITY_ONLY_NO_PROMOTION_AUTHORITY",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "pre_update_checkpoint_sha256": pre,
        "post_update_checkpoint_sha256": post,
        "policy": asdict(policy),
        "retention": {
            "examples": len(retention_group_sha256),
            "groups": len(retention_groups),
            "group_metrics": retention_rows,
            "summary": {
                "before_mean": round(retention_before, 8),
                "after_mean": round(retention_after, 8),
                "overall_regression": round(
                    retention_after - retention_before,
                    8,
                ),
                "mean_group_regression": round(retention_mean, 8),
                "worst_group_regression": round(retention_worst, 8),
                "best_group_regression": round(retention_best, 8),
                "group_regression_stddev": round(retention_dispersion, 8),
            },
        },
        "adaptation": {
            "examples": len(adaptation_group_sha256),
            "groups": len(adaptation_groups),
            "group_metrics": adaptation_rows,
            "summary": {
                "before_mean": round(adaptation_before, 8),
                "after_mean": round(adaptation_after, 8),
                "overall_regression": round(
                    adaptation_after - adaptation_before,
                    8,
                ),
                "mean_group_regression": round(
                    adaptation_mean_regression,
                    8,
                ),
                "mean_group_gain": round(adaptation_mean_gain, 8),
                "worst_group_regression": round(
                    adaptation_worst_regression,
                    8,
                ),
                "best_group_regression": round(
                    adaptation_best_regression,
                    8,
                ),
                "group_regression_stddev": round(
                    adaptation_dispersion,
                    8,
                ),
            },
        },
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_source_group_hash_values": False,
        },
    }
    receipt["court_sha256"] = payload_digest(receipt)
    return receipt


def verify_continual_learning_digest(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported continual-learning court schema")
    supplied = receipt.get("court_sha256")
    body = dict(receipt)
    body.pop("court_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("continual-learning receipt digest mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("continual-learning receipt status invalid")
    return receipt


def verify_continual_learning_receipt(
    receipt: dict[str, Any],
    *,
    pre_update_checkpoint_sha256: str,
    post_update_checkpoint_sha256: str,
    retention_group_sha256: list[str],
    retention_before_values: list[float],
    retention_after_values: list[float],
    adaptation_group_sha256: list[str],
    adaptation_before_values: list[float],
    adaptation_after_values: list[float],
) -> dict[str, Any]:
    verify_continual_learning_digest(receipt)
    policy = ContinualLearningPolicy(**dict(receipt.get("policy", {})))
    computed = assess_continual_learning(
        pre_update_checkpoint_sha256=pre_update_checkpoint_sha256,
        post_update_checkpoint_sha256=post_update_checkpoint_sha256,
        retention_group_sha256=retention_group_sha256,
        retention_before_values=retention_before_values,
        retention_after_values=retention_after_values,
        adaptation_group_sha256=adaptation_group_sha256,
        adaptation_before_values=adaptation_before_values,
        adaptation_after_values=adaptation_after_values,
        policy=policy,
    )
    if computed != receipt:
        raise ValueError("continual-learning receipt does not match evidence")
    return receipt


def effective_continual_promotion_status(
    evaluation: dict[str, Any],
) -> str:
    court = evaluation.get("continual_learning")
    if not isinstance(court, dict):
        return ""
    try:
        verify_continual_learning_digest(court)
    except ValueError:
        return ""
    if court.get("status") != "PASS":
        return ""
    decision = evaluation.get("decision")
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("status", ""))
