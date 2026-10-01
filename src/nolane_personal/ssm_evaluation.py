from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ssm_replacement import BlockReplacementSession, StateSpaceReplacementBlock
from .surgery import parameter_guard_snapshot


@dataclass(slots=True)
class SSMReplacementEvidence:
    test_examples: int
    anchor_examples: int
    baseline_nll: float
    replacement_nll: float
    personal_nll_regression: float
    anchor_baseline_nll: float
    anchor_replacement_nll: float
    anchor_nll_regression: float
    original_layer_parameters: int
    replacement_parameters: int
    parameter_ratio: float
    base_model_unchanged: bool


@dataclass(slots=True)
class SSMReplacementThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    max_personal_nll_regression: float = 0.08
    max_anchor_nll_regression: float = 0.08
    max_parameter_ratio: float = 0.15
    max_replacement_parameters: int = 100_000


@dataclass(slots=True)
class SSMReplacementDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_ssm_replacement(
    evidence: SSMReplacementEvidence,
    thresholds: SSMReplacementThresholds | None = None,
) -> SSMReplacementDecision:
    thresholds = thresholds or SSMReplacementThresholds()
    reasons: list[str] = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.personal_nll_regression > thresholds.max_personal_nll_regression:
        reasons.append("personal_quality_regression")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_quality_regression")
    if evidence.parameter_ratio > thresholds.max_parameter_ratio:
        reasons.append("parameter_ratio_gate_failed")
    if evidence.replacement_parameters > thresholds.max_replacement_parameters:
        reasons.append("replacement_parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    return SSMReplacementDecision(
        status="SSM_REPLACEMENT_PASS" if not reasons else "SSM_REPLACEMENT_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


def _mean_nll(model, encoded_examples, *, replacement=None, layer_index=None) -> float:
    import torch

    device = next(model.parameters()).device
    rows: list[float] = []
    model.eval()
    with torch.no_grad():
        for input_ids, labels, _latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            if replacement is None:
                output = model(input_ids=ids, labels=targets, use_cache=False)
            else:
                replacement.reset_state()
                with BlockReplacementSession(model, int(layer_index), replacement):
                    output = model(input_ids=ids, labels=targets, use_cache=False)
            rows.append(float(output.loss.detach().cpu()) * float(weight))
    return sum(rows) / max(1, len(rows))


def evaluate_ssm_replacement(
    model,
    replacement: StateSpaceReplacementBlock,
    *,
    layer_index: int,
    test_examples,
    anchor_examples,
    original_layer_parameters: int,
    thresholds: SSMReplacementThresholds | None = None,
) -> SSMReplacementDecision:
    before = parameter_guard_snapshot(model)
    baseline = _mean_nll(model, test_examples)
    candidate = _mean_nll(
        model,
        test_examples,
        replacement=replacement,
        layer_index=layer_index,
    )
    anchor_base = _mean_nll(model, anchor_examples)
    anchor_candidate = _mean_nll(
        model,
        anchor_examples,
        replacement=replacement,
        layer_index=layer_index,
    )
    after = parameter_guard_snapshot(model)
    replacement_params = replacement.mixer.parameter_count()
    evidence = SSMReplacementEvidence(
        test_examples=len(test_examples),
        anchor_examples=len(anchor_examples),
        baseline_nll=baseline,
        replacement_nll=candidate,
        personal_nll_regression=candidate - baseline,
        anchor_baseline_nll=anchor_base,
        anchor_replacement_nll=anchor_candidate,
        anchor_nll_regression=anchor_candidate - anchor_base,
        original_layer_parameters=int(original_layer_parameters),
        replacement_parameters=replacement_params,
        parameter_ratio=replacement_params / max(1, int(original_layer_parameters)),
        base_model_unchanged=before == after,
    )
    return decide_ssm_replacement(evidence, thresholds)
