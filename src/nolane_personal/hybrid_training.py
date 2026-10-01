from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .recurrent_cortex import HybridRecurrentCortex
from .surgery import module_parameter_digest, parameter_guard_snapshot


@dataclass(slots=True)
class HybridTrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    initial_effective_gate: float = 0.04

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class HybridTrainingReceipt:
    schema: str
    authority: str
    examples: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_improvement: float
    mixer_parameters: int
    base_model_unchanged: bool
    base_gradients_seen: int
    mixer_gradients_seen: int
    mixer_digest_before: str
    mixer_digest_after: str
    mixer_changed: bool
    effective_gate_before: float
    effective_gate_after: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bootstrap_gate(cortex: HybridRecurrentCortex, target_gate: float) -> None:
    torch = cortex.mixer.torch
    limit = float(cortex.mixer.config.max_abs_gate)
    target = max(-0.95 * limit, min(0.95 * limit, float(target_gate)))
    ratio = target / limit
    raw = 0.5 * math.log((1 + ratio) / (1 - ratio))
    with torch.no_grad():
        cortex.mixer.raw_gate.copy_(
            torch.tensor(
                raw,
                dtype=cortex.mixer.raw_gate.dtype,
                device=cortex.mixer.raw_gate.device,
            )
        )


def _loss(cortex: HybridRecurrentCortex, input_ids, labels, weight: float):
    torch = cortex.mixer.torch
    device = next(cortex.model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    targets = torch.tensor([labels], dtype=torch.long, device=device)
    cortex.reset_recurrent_state()
    outputs = cortex.forward(
        carry_state=False,
        input_ids=ids,
        labels=targets,
        use_cache=False,
    )
    return outputs.loss * float(weight)


def train_hybrid_encoded_examples(
    cortex: HybridRecurrentCortex,
    examples: list[tuple[list[int], list[int], list[float] | None, float]],
    *,
    config: HybridTrainingConfig | None = None,
) -> HybridTrainingReceipt:
    config = config or HybridTrainingConfig()
    config.validate()
    if not examples:
        raise ValueError("no hybrid training examples")

    torch = cortex.mixer.torch
    cortex.assert_gradient_boundary()
    cortex.model.eval()
    cortex.mixer.train()
    device = next(cortex.model.parameters()).device
    cortex.mixer.to(str(device))

    gate_before = float(cortex.mixer.effective_gate().detach().cpu())
    if abs(gate_before) < 1e-8:
        _bootstrap_gate(cortex, config.initial_effective_gate)

    base_before = parameter_guard_snapshot(cortex.model)
    mixer_before = module_parameter_digest(cortex.mixer.module)

    optimizer = torch.optim.AdamW(
        cortex.trainable_parameters(),
        lr=float(config.learning_rate),
    )

    def eval_loss() -> float:
        cortex.mixer.eval()
        rows = []
        with torch.no_grad():
            for input_ids, labels, latent, weight in examples:
                if latent is not None:
                    cortex.set_latent(latent)
                rows.append(float(_loss(cortex, input_ids, labels, weight).detach().cpu()))
        cortex.mixer.train()
        return sum(rows) / len(rows)

    initial_loss = eval_loss()
    best_loss = initial_loss
    steps = 0
    mixer_gradients_seen = 0
    base_gradients_seen = 0

    for _epoch in range(config.epochs):
        for input_ids, labels, latent, weight in examples:
            if latent is not None:
                cortex.set_latent(latent)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(cortex, input_ids, labels, weight)
            loss.backward()
            mixer_gradients_seen += sum(
                1 for p in cortex.mixer.module.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            base_gradients_seen += sum(
                1 for p in cortex.model.parameters() if p.grad is not None
            )
            if base_gradients_seen:
                raise RuntimeError("base Qwen received gradients from hybrid cortex")
            torch.nn.utils.clip_grad_norm_(
                cortex.trainable_parameters(),
                config.max_grad_norm,
            )
            optimizer.step()
            steps += 1
        best_loss = min(best_loss, eval_loss())

    final_loss = eval_loss()
    mixer_after = module_parameter_digest(cortex.mixer.module)
    base_after = parameter_guard_snapshot(cortex.model)

    return HybridTrainingReceipt(
        schema="NOLANE-L7-HYBRID-RECURRENT-TRAINING-V1",
        authority="HYBRID_RECURRENT_CANDIDATE_UNPROMOTED",
        examples=len(examples),
        optimizer_steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_improvement=initial_loss - final_loss,
        mixer_parameters=sum(p.numel() for p in cortex.mixer.module.parameters()),
        base_model_unchanged=base_before == base_after,
        base_gradients_seen=base_gradients_seen,
        mixer_gradients_seen=mixer_gradients_seen,
        mixer_digest_before=mixer_before,
        mixer_digest_after=mixer_after,
        mixer_changed=mixer_before != mixer_after,
        effective_gate_before=float(config.initial_effective_gate),
        effective_gate_after=float(cortex.mixer.effective_gate().detach().cpu()),
        config=asdict(config),
    )
