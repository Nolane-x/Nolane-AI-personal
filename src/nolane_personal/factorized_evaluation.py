from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Any


@dataclass(slots=True)
class FactorizedQualityEvidence:
    test_examples:int
    anchor_examples:int
    rank:int
    hidden_size:int
    dense_boundary_parameters:int
    factorized_boundary_parameters:int
    boundary_parameter_ratio:float
    input_reconstruction_error:float
    output_reconstruction_error:float
    l16_nll:float
    l17_nll:float
    nll_regression_vs_l16:float
    anchor_l16_nll:float
    anchor_l17_nll:float
    anchor_nll_regression:float
    greedy_token_agreement:float
    prompt_scan_equivalence_passed:bool
    cortex_digest_equal_to_l16:bool
    runtime_requires_qwen_model:bool
    runtime_requires_transformers:bool


@dataclass(slots=True)
class FactorizedQualityThresholds:
    min_test_examples:int=2
    min_anchor_examples:int=4
    max_boundary_parameter_ratio:float=0.35
    max_nll_regression_vs_l16:float=0.03
    max_anchor_nll_regression:float=0.05
    min_greedy_token_agreement:float=0.95


def decide_factorized_quality(e,thresholds=None):
    t=thresholds or FactorizedQualityThresholds(); reasons=[]
    if e.test_examples<t.min_test_examples: reasons.append("insufficient_test_examples")
    if e.anchor_examples<t.min_anchor_examples: reasons.append("insufficient_anchor_examples")
    if e.boundary_parameter_ratio>t.max_boundary_parameter_ratio: reasons.append("boundary_compression_failed")
    if e.nll_regression_vs_l16>t.max_nll_regression_vs_l16: reasons.append("l16_noninferiority_failed")
    if e.anchor_nll_regression>t.max_anchor_nll_regression: reasons.append("general_regression_gate_failed")
    if e.greedy_token_agreement<t.min_greedy_token_agreement: reasons.append("greedy_token_agreement_failed")
    if not e.prompt_scan_equivalence_passed: reasons.append("prompt_scan_equivalence_failed")
    if not e.cortex_digest_equal_to_l16: reasons.append("cortex_digest_drift")
    if e.runtime_requires_qwen_model: reasons.append("runtime_qwen_dependency_detected")
    if e.runtime_requires_transformers: reasons.append("runtime_transformers_dependency_detected")
    return {"status":"FACTORIZED_BOUNDARY_QUALITY_PASS" if not reasons else "FACTORIZED_BOUNDARY_QUALITY_BLOCKED","reasons":reasons,"evidence":asdict(e),"thresholds":asdict(t)}


@dataclass(slots=True)
class FactorizedResourceEvidence:
    prompts:int
    l16_median_ms:float
    l17_median_ms:float
    latency_ratio_vs_l16:float
    l16_checkpoint_bytes:int
    l17_checkpoint_bytes:int
    checkpoint_ratio:float
    boundary_parameter_ratio:float
    tokens_per_second:float


@dataclass(slots=True)
class FactorizedResourceThresholds:
    min_prompts:int=4
    max_latency_ratio_vs_l16:float=1.20
    max_checkpoint_ratio:float=0.45
    max_boundary_parameter_ratio:float=0.35
    min_tokens_per_second:float=1.0


def decide_factorized_resources(e,thresholds=None):
    t=thresholds or FactorizedResourceThresholds(); reasons=[]
    if e.prompts<t.min_prompts: reasons.append("insufficient_resource_prompts")
    if e.latency_ratio_vs_l16>t.max_latency_ratio_vs_l16: reasons.append("latency_regression_failed")
    if e.checkpoint_ratio>t.max_checkpoint_ratio: reasons.append("checkpoint_compression_failed")
    if e.boundary_parameter_ratio>t.max_boundary_parameter_ratio: reasons.append("boundary_parameter_compression_failed")
    if e.tokens_per_second<t.min_tokens_per_second: reasons.append("generation_speed_failed")
    return {"status":"FACTORIZED_BOUNDARY_RESOURCE_PASS" if not reasons else "FACTORIZED_BOUNDARY_RESOURCE_BLOCKED","reasons":reasons,"evidence":asdict(e),"thresholds":asdict(t)}
