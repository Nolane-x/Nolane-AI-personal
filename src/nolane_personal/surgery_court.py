from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .surgery import CounterfactualReceipt


@dataclass(slots=True)
class ShadowAdmissionThresholds:
    parameter_cap: int = 100_000
    max_abs_gate: float = 0.10
    max_kl: float = 5.0
    min_cosine_similarity: float = 0.20
    max_overhead_ratio: float = 5.0
    min_receipts: int = 1


@dataclass(slots=True)
class ShadowAdmissionDecision:
    status: str
    reasons: list[str]
    summary: dict[str, Any]
    thresholds: dict[str, Any]


def evaluate_shadow_admission(
    receipts: Iterable[CounterfactualReceipt],
    thresholds: ShadowAdmissionThresholds | None = None,
) -> ShadowAdmissionDecision:
    thresholds = thresholds or ShadowAdmissionThresholds()
    rows = list(receipts)
    reasons: list[str] = []

    if len(rows) < thresholds.min_receipts:
        reasons.append("insufficient_receipts")
        return ShadowAdmissionDecision(
            status="SHADOW_ADMISSION_BLOCKED",
            reasons=reasons,
            summary={"receipts": len(rows)},
            thresholds=asdict(thresholds),
        )

    candidate_ids = {r.candidate_id for r in rows}
    base_ids = {r.base_model_fingerprint for r in rows}
    adapter_ids = {r.adapter_digest for r in rows}
    latent_ids = {r.latent_digest for r in rows}
    if len(candidate_ids) != 1:
        reasons.append("mixed_candidate_ids")
    if len(base_ids) != 1:
        reasons.append("mixed_base_models")
    if len(adapter_ids) != 1:
        reasons.append("mixed_adapter_states")
    if len(latent_ids) != 1:
        reasons.append("mixed_latent_states")
    if any(r.authority != "COUNTERFACTUAL_ONLY_BASELINE_OUTPUT" for r in rows):
        reasons.append("authority_violation")
    if any(not r.base_model_unchanged for r in rows):
        reasons.append("base_model_mutated")
    if any(r.adapter_parameters > thresholds.parameter_cap for r in rows):
        reasons.append("parameter_cap_failed")
    if any(abs(r.gate) > thresholds.max_abs_gate + 1e-12 for r in rows):
        reasons.append("gate_bound_failed")
    if any(not r.layer_indices for r in rows):
        reasons.append("no_layers_injected")

    numeric = [
        r.kl_baseline_to_counterfactual
        for r in rows
    ] + [
        r.mean_abs_logit_shift
        for r in rows
    ] + [
        r.max_abs_logit_shift
        for r in rows
    ] + [
        r.cosine_similarity
        for r in rows
    ] + [
        r.overhead_ratio
        for r in rows
    ]
    if any(not math.isfinite(x) for x in numeric):
        reasons.append("non_finite_metric")

    max_kl = max(r.kl_baseline_to_counterfactual for r in rows)
    min_cosine = min(r.cosine_similarity for r in rows)
    median_overhead = statistics.median(r.overhead_ratio for r in rows)
    if max_kl > thresholds.max_kl:
        reasons.append("counterfactual_kl_exploded")
    if min_cosine < thresholds.min_cosine_similarity:
        reasons.append("counterfactual_cosine_collapsed")
    if median_overhead > thresholds.max_overhead_ratio:
        reasons.append("adapter_overhead_gate_failed")

    summary = {
        "receipts": len(rows),
        "candidate_id": rows[0].candidate_id,
        "base_model_fingerprint": rows[0].base_model_fingerprint,
        "adapter_digest": rows[0].adapter_digest,
        "latent_digest": rows[0].latent_digest,
        "adapter_parameters": max(r.adapter_parameters for r in rows),
        "max_kl": max_kl,
        "median_kl": statistics.median(r.kl_baseline_to_counterfactual for r in rows),
        "max_abs_logit_shift": max(r.max_abs_logit_shift for r in rows),
        "median_abs_logit_shift": statistics.median(r.mean_abs_logit_shift for r in rows),
        "min_cosine_similarity": min_cosine,
        "median_overhead_ratio": median_overhead,
        "top1_change_rate": sum(1 for r in rows if r.top1_changed) / len(rows),
        "base_model_unchanged": all(r.base_model_unchanged for r in rows),
        "authority": "SHADOW_ONLY_NO_PROMOTION",
    }
    return ShadowAdmissionDecision(
        status="SHADOW_ADMISSION_PASS" if not reasons else "SHADOW_ADMISSION_BLOCKED",
        reasons=reasons,
        summary=summary,
        thresholds=asdict(thresholds),
    )
