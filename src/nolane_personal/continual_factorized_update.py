from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
)
from .surgery import module_parameter_digest


@dataclass(slots=True)
class ContinualFactorizedUpdateConfig:
    epochs: int = 2
    learning_rate: float = 5e-4
    adaptation_task_weight: float = 1.0
    retention_task_weight: float = 0.5
    retention_distill_weight: float = 1.0
    distill_temperature: float = 2.0
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >=1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.adaptation_task_weight <= 0:
            raise ValueError("adaptation_task_weight must be positive")
        if self.retention_task_weight < 0:
            raise ValueError("retention_task_weight must be non-negative")
        if self.retention_distill_weight < 0:
            raise ValueError("retention_distill_weight must be non-negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class ContinualFactorizedUpdateReceipt:
    schema: str
    authority: str
    adaptation_examples: int
    retention_examples: int
    epochs: int
    optimizer_steps: int
    adaptation_nll_before: float
    adaptation_nll_after: float
    retention_nll_before: float
    retention_nll_after: float
    candidate_boundary_digest_before: str
    candidate_boundary_digest_after: str
    candidate_boundary_changed: bool
    reference_boundary_digest_before: str
    reference_boundary_digest_after: str
    reference_boundary_unchanged: bool
    candidate_cortex_digest_before: str
    candidate_cortex_digest_after: str
    candidate_cortex_unchanged: bool
    reference_cortex_digest_before: str
    reference_cortex_digest_after: str
    reference_cortex_unchanged: bool
    candidate_boundary_gradients_seen: int
    candidate_cortex_gradients_seen: int
    config: dict[str, Any]
    continual_learning: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _per_example_nll(model, examples) -> list[float]:
    torch = model.cortex.torch
    device = next(model.boundary.module.parameters()).device
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for ids, labels, latent, weight in examples:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            loss = model.forward(input_ids=x, labels=y, state=None).loss
            values.append(float(loss.detach().cpu()) * float(weight))
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _kl(student_logits, teacher_logits, temperature: float):
    torch = __import__("torch")
    t = float(temperature)
    student = torch.nn.functional.log_softmax(student_logits.float() / t, dim=-1)
    teacher = torch.nn.functional.softmax(teacher_logits.float() / t, dim=-1)
    return torch.nn.functional.kl_div(
        student,
        teacher,
        reduction="batchmean",
    ) * (t * t)


def train_continual_factorized_update(
    candidate,
    reference,
    adaptation_examples,
    retention_examples,
    *,
    adaptation_group_sha256: list[str],
    retention_group_sha256: list[str],
    config: ContinualFactorizedUpdateConfig | None = None,
    court_policy: ContinualLearningPolicy | None = None,
) -> ContinualFactorizedUpdateReceipt:
    config = config or ContinualFactorizedUpdateConfig()
    config.validate()
    court_policy = court_policy or ContinualLearningPolicy()

    if candidate is reference:
        raise ValueError("candidate and reference must be distinct model objects")
    if not adaptation_examples:
        raise ValueError("adaptation examples are empty")
    if not retention_examples:
        raise ValueError("retention examples are empty")
    if len(adaptation_group_sha256) != len(adaptation_examples):
        raise ValueError("adaptation source-group lineage length mismatch")
    if len(retention_group_sha256) != len(retention_examples):
        raise ValueError("retention source-group lineage length mismatch")

    torch = candidate.cortex.torch
    device = next(candidate.boundary.module.parameters()).device

    candidate_boundary_before = module_parameter_digest(candidate.boundary.module)
    reference_boundary_before = module_parameter_digest(reference.boundary.module)
    candidate_cortex_before = module_parameter_digest(candidate.cortex.module)
    reference_cortex_before = module_parameter_digest(reference.cortex.module)

    if candidate_boundary_before != reference_boundary_before:
        raise ValueError("candidate must start from the exact reference boundary state")
    if candidate_cortex_before != reference_cortex_before:
        raise ValueError("candidate must start from the exact reference cortex state")

    for parameter in reference.boundary.module.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.cortex.module.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.cortex.module.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.boundary.module.parameters():
        parameter.requires_grad_(True)
    for parameter in candidate.boundary.final_norm.parameters():
        parameter.requires_grad_(False)

    retention_before_values = _per_example_nll(reference, retention_examples)
    adaptation_before_values = _per_example_nll(reference, adaptation_examples)

    params = [
        parameter
        for parameter in candidate.boundary.module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    steps = 0
    boundary_gradients_seen = 0
    cortex_gradients_seen = 0
    reference.eval()
    candidate.eval()

    for _ in range(config.epochs):
        for index, adaptation in enumerate(adaptation_examples):
            a_ids, a_labels, a_latent, a_weight = adaptation
            retention = retention_examples[index % len(retention_examples)]
            r_ids, r_labels, r_latent, r_weight = retention

            ax = torch.tensor([a_ids], dtype=torch.long, device=device)
            ay = torch.tensor([a_labels], dtype=torch.long, device=device)
            rx = torch.tensor([r_ids], dtype=torch.long, device=device)
            ry = torch.tensor([r_labels], dtype=torch.long, device=device)

            if a_latent is not None:
                candidate.set_latent(a_latent)
            adaptation_out = candidate.forward(
                input_ids=ax,
                labels=ay,
                state=None,
            )

            if r_latent is not None:
                candidate.set_latent(r_latent)
                reference.set_latent(r_latent)
            with torch.no_grad():
                reference_logits = reference.forward(
                    input_ids=rx,
                    state=None,
                ).logits.detach()
            retention_out = candidate.forward(
                input_ids=rx,
                labels=ry,
                state=None,
            )
            retention_kl = _kl(
                retention_out.logits,
                reference_logits,
                config.distill_temperature,
            )

            loss = (
                config.adaptation_task_weight
                * adaptation_out.loss
                * float(a_weight)
                + config.retention_task_weight
                * retention_out.loss
                * float(r_weight)
                + config.retention_distill_weight
                * retention_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            cortex_gradients_seen += sum(
                1
                for parameter in candidate.cortex.module.parameters()
                if parameter.grad is not None
            )
            if cortex_gradients_seen:
                raise RuntimeError(
                    "candidate cortex received gradients during continual boundary update"
                )
            boundary_gradients_seen += sum(
                1
                for parameter in params
                if parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
            )
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimizer.step()
            steps += 1

    retention_after_values = _per_example_nll(candidate, retention_examples)
    adaptation_after_values = _per_example_nll(candidate, adaptation_examples)

    candidate_boundary_after = module_parameter_digest(candidate.boundary.module)
    reference_boundary_after = module_parameter_digest(reference.boundary.module)
    candidate_cortex_after = module_parameter_digest(candidate.cortex.module)
    reference_cortex_after = module_parameter_digest(reference.cortex.module)

    continual = assess_continual_learning(
        pre_update_checkpoint_sha256=reference_boundary_before,
        post_update_checkpoint_sha256=candidate_boundary_after,
        retention_group_sha256=retention_group_sha256,
        retention_before_values=retention_before_values,
        retention_after_values=retention_after_values,
        adaptation_group_sha256=adaptation_group_sha256,
        adaptation_before_values=adaptation_before_values,
        adaptation_after_values=adaptation_after_values,
        policy=court_policy,
    )

    return ContinualFactorizedUpdateReceipt(
        schema="NOLANE-L31-FACTORIZED-CONTINUAL-UPDATE-V1",
        authority="CONTINUAL_FACTORIZED_UPDATE_CANDIDATE_ONLY",
        adaptation_examples=len(adaptation_examples),
        retention_examples=len(retention_examples),
        epochs=config.epochs,
        optimizer_steps=steps,
        adaptation_nll_before=_mean(adaptation_before_values),
        adaptation_nll_after=_mean(adaptation_after_values),
        retention_nll_before=_mean(retention_before_values),
        retention_nll_after=_mean(retention_after_values),
        candidate_boundary_digest_before=candidate_boundary_before,
        candidate_boundary_digest_after=candidate_boundary_after,
        candidate_boundary_changed=(
            candidate_boundary_before != candidate_boundary_after
        ),
        reference_boundary_digest_before=reference_boundary_before,
        reference_boundary_digest_after=reference_boundary_after,
        reference_boundary_unchanged=(
            reference_boundary_before == reference_boundary_after
        ),
        candidate_cortex_digest_before=candidate_cortex_before,
        candidate_cortex_digest_after=candidate_cortex_after,
        candidate_cortex_unchanged=(
            candidate_cortex_before == candidate_cortex_after
        ),
        reference_cortex_digest_before=reference_cortex_before,
        reference_cortex_digest_after=reference_cortex_after,
        reference_cortex_unchanged=(
            reference_cortex_before == reference_cortex_after
        ),
        candidate_boundary_gradients_seen=boundary_gradients_seen,
        candidate_cortex_gradients_seen=cortex_gradients_seen,
        config=asdict(config),
        continual_learning=continual,
    )
