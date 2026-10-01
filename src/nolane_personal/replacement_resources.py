from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ReplacementResourceEvidence:
    prompts: int
    total_decoder_layers: int
    skipped_layers: int
    baseline_median_ms: float
    replacement_median_ms: float
    latency_ratio_vs_base: float
    artifact_bytes: int
    replacement_parameters: int


@dataclass(slots=True)
class ReplacementResourceThresholds:
    min_prompts: int = 4
    min_skipped_layers: int = 1
    max_latency_ratio_vs_base: float = 0.95
    max_artifact_bytes: int = 2_000_000
    parameter_cap: int = 100_000


def decide_replacement_resources(evidence: ReplacementResourceEvidence, thresholds=None):
    thresholds = thresholds or ReplacementResourceThresholds()
    reasons = []
    if evidence.prompts < thresholds.min_prompts:
        reasons.append("insufficient_resource_prompts")
    if evidence.skipped_layers < thresholds.min_skipped_layers:
        reasons.append("no_decoder_blocks_skipped")
    if evidence.skipped_layers >= evidence.total_decoder_layers:
        reasons.append("cannot_replace_all_decoder_layers")
    if evidence.baseline_median_ms <= 0 or evidence.replacement_median_ms <= 0:
        reasons.append("invalid_latency_measurement")
    if evidence.latency_ratio_vs_base > thresholds.max_latency_ratio_vs_base:
        reasons.append("latency_gain_gate_failed")
    if evidence.artifact_bytes > thresholds.max_artifact_bytes:
        reasons.append("artifact_size_gate_failed")
    if evidence.replacement_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    return {
        "status": "BLOCK_REPLACEMENT_RESOURCE_PASS" if not reasons else "BLOCK_REPLACEMENT_RESOURCE_BLOCKED",
        "reasons": reasons,
        "evidence": asdict(evidence),
        "thresholds": asdict(thresholds),
    }
