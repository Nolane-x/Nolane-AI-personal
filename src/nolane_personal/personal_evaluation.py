from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .personal_cortex import TrainablePersonalCortex


@dataclass(slots=True)
class PersonalQualityEvidence:
    test_examples: int
    baseline_nll: float
    personal_nll: float
    nll_improvement: float
    adapter_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int


@dataclass(slots=True)
class PersonalQualityThresholds:
    min_test_examples: int = 2
    min_nll_improvement: float = 0.01
    parameter_cap: int = 100_000


@dataclass(slots=True)
class PersonalQualityDecision:
    status: str
    reasons: list[str]
    evidence: dict[str, Any]
    thresholds: dict[str, Any]


def decide_personal_quality(
    evidence: PersonalQualityEvidence,
    thresholds: PersonalQualityThresholds | None = None,
) -> PersonalQualityDecision:
    thresholds = thresholds or PersonalQualityThresholds()
    reasons: list[str] = []
    if evidence.test_examples < thresholds.min_test_examples:
        reasons.append("insufficient_test_examples")
    if evidence.nll_improvement < thresholds.min_nll_improvement:
        reasons.append("heldout_nll_gate_failed")
    if evidence.adapter_parameters > thresholds.parameter_cap:
        reasons.append("parameter_cap_failed")
    if not evidence.base_model_unchanged:
        reasons.append("base_model_mutated")
    if evidence.base_gradients_seen != 0:
        reasons.append("base_gradient_boundary_failed")
    return PersonalQualityDecision(
        status="PERSONAL_QUALITY_PASS" if not reasons else "PERSONAL_QUALITY_BLOCKED",
        reasons=reasons,
        evidence=asdict(evidence),
        thresholds=asdict(thresholds),
    )


def mean_encoded_nll(
    cortex: TrainablePersonalCortex,
    encoded_examples: list[tuple[list[int], list[int], list[float] | None, float]],
    *,
    personalized: bool,
) -> float:
    torch = cortex.adapter.torch
    device = next(cortex.model.parameters()).device
    losses: list[float] = []
    cortex.model.eval()
    cortex.adapter.eval()

    with torch.no_grad():
        for input_ids, labels, latent, weight in encoded_examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                cortex.set_latent(latent)
            if personalized:
                outputs = cortex.forward(input_ids=ids, labels=y, use_cache=False)
            else:
                outputs = cortex.model(input_ids=ids, labels=y, use_cache=False)
            losses.append(float(outputs.loss.detach().cpu()) * float(weight))
    return sum(losses) / max(1, len(losses))
