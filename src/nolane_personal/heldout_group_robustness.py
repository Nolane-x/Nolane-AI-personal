from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import pstdev
from typing import Any

from .personal_protocol import GROUPED_SPLIT_STRATEGY
from .store import payload_digest


SCHEMA = "NOLANE-L29-HELDOUT-GROUP-ROBUSTNESS-V1"


@dataclass(slots=True)
class HeldoutGroupRobustnessPolicy:
    min_groups: int = 2
    max_worst_group_regression: float = 0.03

    def validate(self) -> None:
        if self.min_groups < 2:
            raise ValueError("min_groups must be >=2")
        if self.max_worst_group_regression < 0:
            raise ValueError("max_worst_group_regression must be non-negative")


def assess_group_robustness(
    protocol: dict[str, Any],
    *,
    split: str,
    reference_values: list[float],
    candidate_values: list[float],
    policy: HeldoutGroupRobustnessPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or HeldoutGroupRobustnessPolicy()
    policy.validate()
    if split not in {"dev", "test"}:
        raise ValueError("group robustness split must be dev or test")
    rows = list(protocol.get("splits", {}).get(split, []))
    if len(reference_values) != len(rows) or len(candidate_values) != len(rows):
        raise ValueError("group robustness metric length does not match protocol split")
    if any(not math.isfinite(float(value)) for value in reference_values + candidate_values):
        raise ValueError("group robustness metrics must be finite")

    reasons: list[str] = []
    if protocol.get("split_strategy") != GROUPED_SPLIT_STRATEGY:
        reasons.append("non_grouped_split_strategy")

    grouped: dict[str, list[int]] = {}
    group_order: list[str] = []
    missing = 0
    for local_index, row in enumerate(rows):
        group = row.get("source_group_sha256")
        if not isinstance(group, str) or not group:
            missing += 1
            continue
        if group not in grouped:
            grouped[group] = []
            group_order.append(group)
        grouped[group].append(local_index)

    if missing:
        reasons.append("missing_source_group_lineage")
    if len(group_order) < policy.min_groups:
        reasons.append("insufficient_heldout_source_groups")

    group_metrics: list[dict[str, Any]] = []
    regressions: list[float] = []
    for ordinal, group in enumerate(group_order):
        indices = grouped[group]
        ref = sum(float(reference_values[i]) for i in indices) / len(indices)
        cand = sum(float(candidate_values[i]) for i in indices) / len(indices)
        regression = cand - ref
        regressions.append(regression)
        group_metrics.append({
            "group_alias": f"group_{ordinal:03d}",
            "examples": len(indices),
            "split_local_indices": list(indices),
            "reference_mean": round(ref, 8),
            "candidate_mean": round(cand, 8),
            "regression": round(regression, 8),
        })

    overall_reference = (
        sum(float(value) for value in reference_values) / len(reference_values)
        if reference_values else 0.0
    )
    overall_candidate = (
        sum(float(value) for value in candidate_values) / len(candidate_values)
        if candidate_values else 0.0
    )
    worst = max(regressions) if regressions else float("inf")
    best = min(regressions) if regressions else float("inf")
    mean_group = sum(regressions) / len(regressions) if regressions else float("inf")
    dispersion = pstdev(regressions) if len(regressions) >= 2 else 0.0

    if regressions and worst > policy.max_worst_group_regression:
        reasons.append("worst_group_noninferiority_failed")

    receipt = {
        "schema": SCHEMA,
        "authority": "HELDOUT_GROUP_ROBUSTNESS_ONLY_NO_PROMOTION_AUTHORITY",
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "protocol_sha256": str(protocol.get("protocol_sha256")),
        "split": split,
        "policy": asdict(policy),
        "examples": len(rows),
        "groups": len(group_order),
        "group_metrics": group_metrics,
        "summary": {
            "overall_reference_mean": round(overall_reference, 8),
            "overall_candidate_mean": round(overall_candidate, 8),
            "overall_regression": round(overall_candidate-overall_reference, 8),
            "mean_group_regression": round(mean_group, 8) if math.isfinite(mean_group) else None,
            "worst_group_regression": round(worst, 8) if math.isfinite(worst) else None,
            "best_group_regression": round(best, 8) if math.isfinite(best) else None,
            "group_regression_stddev": round(dispersion, 8),
            "groups_with_positive_regression": sum(1 for value in regressions if value > 0),
        },
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_source_group_hash_values": False,
        },
    }
    receipt["court_sha256"] = payload_digest(receipt)
    return receipt


def verify_group_robustness_receipt(
    receipt: dict[str, Any],
    *,
    protocol: dict[str, Any],
    reference_values: list[float],
    candidate_values: list[float],
) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported heldout group robustness schema")
    supplied = receipt.get("court_sha256")
    body = dict(receipt)
    body.pop("court_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("heldout group robustness receipt digest mismatch")
    policy = HeldoutGroupRobustnessPolicy(**dict(receipt.get("policy", {})))
    computed = assess_group_robustness(
        protocol,
        split=str(receipt.get("split")),
        reference_values=reference_values,
        candidate_values=candidate_values,
        policy=policy,
    )
    if computed != receipt:
        raise ValueError("heldout group robustness receipt does not match evidence")
    return receipt


def verify_group_robustness_digest(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported heldout group robustness schema")
    supplied = receipt.get("court_sha256")
    body = dict(receipt)
    body.pop("court_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("heldout group robustness receipt digest mismatch")
    if receipt.get("status") not in {"PASS", "BLOCKED"}:
        raise ValueError("heldout group robustness status invalid")
    return receipt


def effective_quality_status(evaluation: dict[str, Any]) -> str:
    group = evaluation.get("group_robustness")
    if not isinstance(group, dict):
        return ""
    try:
        verify_group_robustness_digest(group)
    except ValueError:
        return ""
    if group.get("status") != "PASS":
        return ""
    decision = evaluation.get("decision")
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("status", ""))
