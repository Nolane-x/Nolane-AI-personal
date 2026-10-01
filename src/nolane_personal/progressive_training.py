from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .replacement_training import (
    ReplacementTrainingConfig,
    mean_encoded_nll,
    train_encoded_examples,
)
from .surgery import module_parameter_digest, parameter_guard_snapshot


@dataclass(slots=True)
class ProgressiveTrainingConfig:
    distill_epochs_per_stage: int = 1
    task_epochs_per_stage: int = 2
    learning_rate: float = 2e-3
    distill_learning_rate: float = 3e-3
    max_grad_norm: float = 1.0
    max_dev_regression: float = 0.05

    def validate(self) -> None:
        if self.distill_epochs_per_stage < 0:
            raise ValueError("distill_epochs_per_stage must be >=0")
        if self.task_epochs_per_stage < 1:
            raise ValueError("task_epochs_per_stage must be >=1")
        if self.learning_rate <= 0 or self.distill_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.max_dev_regression < 0:
            raise ValueError("max_dev_regression must be >=0")


@dataclass(slots=True)
class ProgressiveStageReceipt:
    stage: int
    layer_indices: list[int]
    replaced_fraction: float
    train_loss_before: float
    train_loss_after: float
    dev_nll_before: float
    dev_nll_after: float
    dev_regression_vs_previous: float
    accepted: bool
    rollback_reason: str | None
    replacement_digest_before: str
    replacement_digest_after: str
    optimizer_steps: int
    base_model_unchanged: bool
    base_gradients_seen: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProgressiveTrainingReceipt:
    schema: str
    authority: str
    plan_sha256: str
    stages_attempted: int
    stages_accepted: int
    final_layer_indices: list[int]
    final_replaced_fraction: float
    replacement_parameters: int
    replacement_digest_before: str
    replacement_digest_after: str
    replacement_changed: bool
    base_model_unchanged: bool
    base_gradients_seen: int
    stage_receipts: list[dict[str, Any]]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_state(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def train_progressive_stages(
    cortex,
    train_examples,
    dev_examples,
    plan: dict[str, Any],
    *,
    config: ProgressiveTrainingConfig | None = None,
) -> ProgressiveTrainingReceipt:
    config = config or ProgressiveTrainingConfig()
    config.validate()
    if not train_examples:
        raise ValueError("progressive train examples are empty")
    if not dev_examples:
        raise ValueError("progressive dev examples are empty")
    stages = list(plan.get("stages", []))
    if not stages:
        raise ValueError("progressive plan has no stages")

    base_before = parameter_guard_snapshot(cortex.model)
    digest_before = module_parameter_digest(cortex.replacement.module)
    accepted_layers: tuple[int, ...] = tuple()
    accepted_dev = None
    stage_receipts: list[dict[str, Any]] = []
    base_gradients_seen = 0

    for row in stages:
        stage_number = int(row["stage"])
        layers = tuple(int(x) for x in row["layer_indices"])
        if accepted_layers and not set(accepted_layers).issubset(layers):
            raise ValueError("progressive plan attempted to remove an accepted layer")

        snapshot = _clone_state(cortex.replacement.module)
        previous_layers = cortex.config.layer_indices
        cortex.config.layer_indices = layers
        dev_before = mean_encoded_nll(cortex, dev_examples, replacement_enabled=True)
        stage_digest_before = module_parameter_digest(cortex.replacement.module)

        receipt = train_encoded_examples(
            cortex,
            train_examples,
            config=ReplacementTrainingConfig(
                distill_epochs=config.distill_epochs_per_stage,
                task_epochs=config.task_epochs_per_stage,
                learning_rate=config.learning_rate,
                distill_learning_rate=config.distill_learning_rate,
                max_grad_norm=config.max_grad_norm,
            ),
        )
        dev_after = mean_encoded_nll(cortex, dev_examples, replacement_enabled=True)
        reference = accepted_dev if accepted_dev is not None else dev_before
        regression = dev_after - reference
        accepted = bool(receipt.base_model_unchanged and receipt.base_gradients_seen == 0)
        reason = None
        if regression > config.max_dev_regression:
            accepted = False
            reason = "dev_regression_gate_failed"

        if accepted:
            accepted_layers = layers
            accepted_dev = dev_after
        else:
            cortex.replacement.module.load_state_dict(snapshot)
            cortex.config.layer_indices = accepted_layers or previous_layers

        base_gradients_seen += int(receipt.base_gradients_seen)
        stage_receipts.append(
            ProgressiveStageReceipt(
                stage=stage_number,
                layer_indices=list(layers),
                replaced_fraction=float(row["replaced_fraction"]),
                train_loss_before=float(receipt.task_initial_loss),
                train_loss_after=float(receipt.task_final_loss),
                dev_nll_before=float(dev_before),
                dev_nll_after=float(dev_after),
                dev_regression_vs_previous=float(regression),
                accepted=accepted,
                rollback_reason=reason,
                replacement_digest_before=stage_digest_before,
                replacement_digest_after=module_parameter_digest(cortex.replacement.module),
                optimizer_steps=int(receipt.optimizer_steps),
                base_model_unchanged=bool(receipt.base_model_unchanged),
                base_gradients_seen=int(receipt.base_gradients_seen),
            ).to_dict()
        )
        if not accepted:
            break

    if not accepted_layers:
        raise RuntimeError("no progressive replacement stage passed the dev gate")

    cortex.config.layer_indices = accepted_layers
    base_after = parameter_guard_snapshot(cortex.model)
    digest_after = module_parameter_digest(cortex.replacement.module)
    return ProgressiveTrainingReceipt(
        schema="NOLANE-L10-PROGRESSIVE-REPLACEMENT-TRAINING-V1",
        authority="TRAINED_PROGRESSIVE_CANDIDATE_ONLY",
        plan_sha256=str(plan["plan_sha256"]),
        stages_attempted=len(stage_receipts),
        stages_accepted=sum(1 for row in stage_receipts if row["accepted"]),
        final_layer_indices=list(accepted_layers),
        final_replaced_fraction=len(accepted_layers) / int(plan["total_layers"]),
        replacement_parameters=cortex.trainable_parameter_count(),
        replacement_digest_before=digest_before,
        replacement_digest_after=digest_after,
        replacement_changed=digest_before != digest_after,
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        stage_receipts=stage_receipts,
        config=asdict(config),
    )
