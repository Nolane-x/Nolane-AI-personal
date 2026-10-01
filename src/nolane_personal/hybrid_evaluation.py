from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .recurrent_cortex import HybridRecurrentCortex
from .surgery import parameter_guard_snapshot


@dataclass(slots=True)
class HybridQualityEvidence:
    test_examples: int
    anchor_examples: int
    baseline_nll: float
    hybrid_nll: float
    nll_improvement: float
    anchor_baseline_nll: float
    anchor_hybrid_nll: float
    anchor_nll_regression: float
    mixer_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class HybridQualityThresholds:
    min_test_examples: int = 2
    min_anchor_examples: int = 4
    min_nll_improvement: float = 0.01
    max_anchor_nll_regression: float = 0.05
    parameter_cap: int = 100_000


@dataclass(slots=True)
class HybridQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_hybrid_quality(
    evidence: HybridQualityEvidence,
    thresholds: HybridQualityThresholds | None = None,
) -> HybridQualityDecision:
    thresholds = thresholds or HybridQualityThresholds()
    reasons: list[str] = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.anchor_examples < thresholds.min_anchor_examples:
        reasons.append("insufficient_anchor_examples")
    if evidence.nll_improvement < thresholds.min_nll_improvement:
        reasons.append("heldout_nll_gate_failed")
    if evidence.anchor_nll_regression > thresholds.max_anchor_nll_regression:
        reasons.append("general_regression_gate_failed")
    if evidence.mixer_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen:
        reasons.append("base_gradient_boundary_failed")
    return HybridQualityDecision(
        status="HYBRID_QUALITY_PASS" if not reasons else "HYBRID_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


def mean_hybrid_nll(
    cortex: HybridRecurrentCortex,
    examples: list[tuple[list[int], list[int], list[float] | None, float]],
    *,
    hybrid: bool,
) -> float:
    torch = cortex.mixer.torch
    device = next(cortex.model.parameters()).device
    rows: list[float] = []
    cortex.model.eval()
    cortex.mixer.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in examples:
            if latent is not None:
                cortex.set_latent(latent)
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            cortex.reset_recurrent_state()
            if hybrid:
                output = cortex.forward(
                    carry_state=False,
                    input_ids=ids,
                    labels=targets,
                    use_cache=False,
                )
            else:
                output = cortex.model(
                    input_ids=ids,
                    labels=targets,
                    use_cache=False,
                )
            rows.append(float(output.loss.detach().cpu()) * float(weight))
    return sum(rows) / max(1, len(rows))


def evaluate_hybrid(
    cortex: HybridRecurrentCortex,
    test_examples,
    anchor_examples,
) -> HybridQualityDecision:
    before = parameter_guard_snapshot(cortex.model)
    baseline = mean_hybrid_nll(cortex, test_examples, hybrid=False)
    hybrid = mean_hybrid_nll(cortex, test_examples, hybrid=True)
    anchor_base = mean_hybrid_nll(cortex, anchor_examples, hybrid=False)
    anchor_hybrid = mean_hybrid_nll(cortex, anchor_examples, hybrid=True)
    after = parameter_guard_snapshot(cortex.model)
    evidence = HybridQualityEvidence(
        test_examples=len(test_examples),
        anchor_examples=len(anchor_examples),
        baseline_nll=baseline,
        hybrid_nll=hybrid,
        nll_improvement=baseline - hybrid,
        anchor_baseline_nll=anchor_base,
        anchor_hybrid_nll=anchor_hybrid,
        anchor_nll_regression=anchor_hybrid - anchor_base,
        mixer_parameters=sum(p.numel() for p in cortex.mixer.module.parameters()),
        base_model_unchanged=before == after,
        base_gradients_seen=sum(1 for p in cortex.model.parameters() if p.grad is not None),
    )
    return decide_hybrid_quality(evidence)
