from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .deep_recurrent_cortex import DeepRecurrentCortexConfig, DeepRecurrentStateSpaceCortex
from .factorized_boundary import factorize_standalone_boundary
from .factorized_training import FactorizedTrainingConfig, train_factorized_boundary
from .standalone_model import StandaloneNolaneLM
from .surgery import module_parameter_digest


@dataclass(slots=True)
class RankFrontierConfig:
    ranks: tuple[int, ...] = (256, 192, 128, 96, 64)
    max_dev_nll_regression_per_step: float = 0.02
    max_dev_nll_regression_vs_l16: float = 0.05
    min_greedy_token_agreement: float = 0.93
    max_boundary_parameter_ratio: float = 0.60
    distill_epochs: int = 3
    learning_rate: float = 2e-3
    distill_weight: float = 1.0
    distill_temperature: float = 2.0

    def validate(self, *, hidden_size: int) -> None:
        if not self.ranks:
            raise ValueError("rank frontier cannot be empty")
        previous = hidden_size + 1
        for rank in self.ranks:
            if rank < 1 or rank > hidden_size:
                raise ValueError("rank frontier entry out of range")
            if rank >= previous:
                raise ValueError("rank frontier must be strictly descending")
            previous = rank
        if self.max_dev_nll_regression_per_step < 0:
            raise ValueError("max_dev_nll_regression_per_step must be >=0")
        if self.max_dev_nll_regression_vs_l16 < 0:
            raise ValueError("max_dev_nll_regression_vs_l16 must be >=0")
        if not 0.0 <= self.min_greedy_token_agreement <= 1.0:
            raise ValueError("min_greedy_token_agreement must be in [0,1]")
        if not 0.0 < self.max_boundary_parameter_ratio < 1.0:
            raise ValueError("max_boundary_parameter_ratio must be in (0,1)")
        if self.distill_epochs < 1:
            raise ValueError("distill_epochs must be >=1")


@dataclass(slots=True)
class RankStageReceipt:
    rank: int
    boundary_parameters: int
    boundary_parameter_ratio: float
    reconstruction_input_error: float
    reconstruction_output_error: float
    train_nll_before: float
    train_nll_after: float
    dev_nll: float
    dev_regression_vs_l16: float
    dev_regression_vs_previous: float
    greedy_token_agreement: float
    accepted: bool
    reason: str | None
    cortex_digest: str
    cortex_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RankFrontierReceipt:
    schema: str
    authority: str
    source_boundary_parameters: int
    source_cortex_digest: str
    l16_dev_nll: float
    ranks_attempted: list[int]
    ranks_accepted: list[int]
    selected_rank: int
    selected_boundary_parameters: int
    selected_boundary_parameter_ratio: float
    selected_dev_nll: float
    stages: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_cortex(source, *, device: str):
    cortex = DeepRecurrentStateSpaceCortex(
        source.cortex.hidden_size,
        DeepRecurrentCortexConfig(**asdict(source.cortex.config)),
        seed=0,
    )
    cortex.module.load_state_dict(source.cortex.module.state_dict())
    cortex.to(device).eval()
    return cortex


def _nll_and_predictions(model, examples):
    torch = model.cortex.torch
    device = next(model.boundary.module.parameters()).device
    losses = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for input_ids, labels, latent, weight in examples:
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            target = torch.tensor([labels], dtype=torch.long, device=device)
            if latent is not None:
                model.set_latent(latent)
            output = model.forward(input_ids=ids, labels=target, state=None)
            losses.append(float(output.loss.detach().cpu()) * float(weight))
            predictions.append(torch.argmax(output.logits, dim=-1).detach().cpu())
    return sum(losses) / max(1, len(losses)), predictions


def _agreement(reference, candidate) -> float:
    total = 0
    equal = 0
    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            raise ValueError("rank-frontier prediction shape mismatch")
        total += left.numel()
        equal += int((left == right).sum())
    return equal / max(1, total)


def search_rank_frontier(
    source_l16,
    train_examples,
    dev_examples,
    *,
    config: RankFrontierConfig | None = None,
    device: str | None = None,
):
    if not train_examples or not dev_examples:
        raise ValueError("rank frontier requires non-empty train/dev evidence")
    hidden_size = int(source_l16.config.hidden_size)
    config = config or RankFrontierConfig(
        ranks=tuple(rank for rank in (256, 192, 128, 96, 64) if rank <= hidden_size)
    )
    config.validate(hidden_size=hidden_size)
    if device is None:
        device = str(next(source_l16.boundary.module.parameters()).device)

    source_l16.eval()
    l16_dev_nll, l16_predictions = _nll_and_predictions(source_l16, dev_examples)
    source_cortex_digest = module_parameter_digest(source_l16.cortex.module)
    source_boundary_parameters = source_l16.boundary.parameter_count()

    accepted_model = None
    accepted_rank = None
    accepted_nll = l16_dev_nll
    accepted_boundary_parameters = source_boundary_parameters
    accepted_ratio = 1.0
    stages = []

    for rank in config.ranks:
        boundary, factor_receipt = factorize_standalone_boundary(
            source_l16.boundary,
            rank=rank,
            seed=rank,
        )
        boundary.to(device)
        cortex = _clone_cortex(source_l16, device=device)
        candidate = StandaloneNolaneLM(
            boundary,
            cortex,
            source_l16.latent,
            carry_recurrent_state=source_l16.carry_recurrent_state,
        ).to(device)

        training = train_factorized_boundary(
            candidate,
            source_l16,
            train_examples,
            config=FactorizedTrainingConfig(
                epochs=config.distill_epochs,
                learning_rate=config.learning_rate,
                distill_weight=config.distill_weight,
                distill_temperature=config.distill_temperature,
            ),
        )
        dev_nll, candidate_predictions = _nll_and_predictions(candidate, dev_examples)
        agreement = _agreement(l16_predictions, candidate_predictions)
        regression_base = dev_nll - l16_dev_nll
        regression_previous = dev_nll - accepted_nll
        cortex_digest = module_parameter_digest(candidate.cortex.module)
        cortex_unchanged = cortex_digest == source_cortex_digest

        reason = None
        if float(factor_receipt["parameter_ratio"]) > config.max_boundary_parameter_ratio:
            reason = "boundary_compression_gate_failed"
        elif not cortex_unchanged:
            reason = "cortex_digest_drift"
        elif regression_base > config.max_dev_nll_regression_vs_l16:
            reason = "l16_dev_regression_gate_failed"
        elif regression_previous > config.max_dev_nll_regression_per_step:
            reason = "step_dev_regression_gate_failed"
        elif agreement < config.min_greedy_token_agreement:
            reason = "greedy_token_agreement_gate_failed"

        accepted = reason is None
        stages.append(
            RankStageReceipt(
                rank=rank,
                boundary_parameters=int(factor_receipt["factorized_parameters"]),
                boundary_parameter_ratio=float(factor_receipt["parameter_ratio"]),
                reconstruction_input_error=float(factor_receipt["input_relative_frobenius_error"]),
                reconstruction_output_error=float(factor_receipt["output_relative_frobenius_error"]),
                train_nll_before=float(training.task_initial_loss),
                train_nll_after=float(training.task_final_loss),
                dev_nll=float(dev_nll),
                dev_regression_vs_l16=float(regression_base),
                dev_regression_vs_previous=float(regression_previous),
                greedy_token_agreement=float(agreement),
                accepted=accepted,
                reason=reason,
                cortex_digest=cortex_digest,
                cortex_unchanged=cortex_unchanged,
            ).to_dict()
        )
        if not accepted:
            break
        accepted_model = candidate
        accepted_rank = rank
        accepted_nll = dev_nll
        accepted_boundary_parameters = int(factor_receipt["factorized_parameters"])
        accepted_ratio = float(factor_receipt["parameter_ratio"])

    if accepted_model is None or accepted_rank is None:
        raise RuntimeError("no low-rank boundary candidate passed the frontier gates")

    receipt = RankFrontierReceipt(
        schema="NOLANE-L18-ADAPTIVE-RANK-FRONTIER-V1",
        authority="RANK_FRONTIER_CANDIDATE_ONLY",
        source_boundary_parameters=source_boundary_parameters,
        source_cortex_digest=source_cortex_digest,
        l16_dev_nll=float(l16_dev_nll),
        ranks_attempted=[int(row["rank"]) for row in stages],
        ranks_accepted=[int(row["rank"]) for row in stages if row["accepted"]],
        selected_rank=int(accepted_rank),
        selected_boundary_parameters=int(accepted_boundary_parameters),
        selected_boundary_parameter_ratio=float(accepted_ratio),
        selected_dev_nll=float(accepted_nll),
        stages=stages,
        config=asdict(config),
    )
    return accepted_model, receipt
