from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .ssm_replacement import SelectiveStateSpaceMixer


@dataclass(slots=True)
class HiddenPair:
    input_hidden: Any
    target_hidden: Any
    latent: list[float]


@dataclass(slots=True)
class SSMTrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    initial_effective_gate: float = 0.25

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >=1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


@dataclass(slots=True)
class SSMDistillReceipt:
    schema: str
    pairs: int
    optimizer_steps: int
    initial_mse: float
    final_mse: float
    best_mse: float
    mse_improvement: float
    replacement_parameters: int
    gradients_seen: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hidden_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def capture_hidden_pairs(
    model,
    layer_index: int,
    input_batches: list[dict[str, Any]],
    latents: list[list[float]],
) -> list[HiddenPair]:
    from .ssm_replacement import resolve_layer_container

    if len(input_batches) != len(latents):
        raise ValueError("input_batches/latents length mismatch")
    container = resolve_layer_container(model)
    layer = container[int(layer_index)]
    captured: list[HiddenPair] = []
    current_input = None

    def pre_hook(_module, args, kwargs):
        nonlocal current_input
        hidden = args[0] if args else kwargs["hidden_states"]
        current_input = hidden.detach().cpu()

    def post_hook(_module, _args, output):
        nonlocal current_input
        if current_input is None:
            raise RuntimeError("missing captured layer input")
        captured.append(
            HiddenPair(
                input_hidden=current_input,
                target_hidden=_hidden_tensor(output).detach().cpu(),
                latent=latents[len(captured)],
            )
        )
        current_input = None

    pre = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post = layer.register_forward_hook(post_hook)
    try:
        model.eval()
        with model.device if False else _nullcontext():
            pass
        import torch
        with torch.inference_mode():
            for batch in input_batches:
                model(**batch, use_cache=False)
    finally:
        post.remove()
        pre.remove()
    return captured


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def _bootstrap_gate(mixer: SelectiveStateSpaceMixer, target: float) -> None:
    torch = mixer.torch
    limit = mixer.config.max_abs_gate
    target = max(-0.95 * limit, min(0.95 * limit, float(target)))
    ratio = target / limit
    raw = 0.5 * math.log((1 + ratio) / (1 - ratio))
    with torch.no_grad():
        mixer.raw_gate.copy_(
            torch.tensor(raw, dtype=mixer.raw_gate.dtype, device=mixer.raw_gate.device)
        )


def distill_hidden_pairs(
    mixer: SelectiveStateSpaceMixer,
    pairs: list[HiddenPair],
    *,
    config: SSMTrainingConfig | None = None,
) -> SSMDistillReceipt:
    config = config or SSMTrainingConfig()
    config.validate()
    if not pairs:
        raise ValueError("no hidden pairs")
    torch = mixer.torch
    device = next(mixer.module.parameters()).device
    mixer.train()

    if abs(float(mixer.effective_gate().detach().cpu())) < 1e-8:
        _bootstrap_gate(mixer, config.initial_effective_gate)

    optimizer = torch.optim.AdamW(mixer.module.parameters(), lr=config.learning_rate)

    def loss_value() -> float:
        mixer.eval()
        rows = []
        with torch.no_grad():
            for pair in pairs:
                x = pair.input_hidden.to(device=device)
                y = pair.target_hidden.to(device=device)
                pred, _ = mixer.scan(x, pair.latent, state=None)
                rows.append(float(torch.nn.functional.mse_loss(pred, y).cpu()))
        mixer.train()
        return sum(rows) / len(rows)

    initial = loss_value()
    best = initial
    gradients_seen = 0
    steps = 0
    for _epoch in range(config.epochs):
        for pair in pairs:
            optimizer.zero_grad(set_to_none=True)
            x = pair.input_hidden.to(device=device)
            y = pair.target_hidden.to(device=device)
            pred, _ = mixer.scan(x, pair.latent, state=None)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            gradients_seen += sum(
                1 for p in mixer.module.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            )
            torch.nn.utils.clip_grad_norm_(mixer.module.parameters(), config.max_grad_norm)
            optimizer.step()
            steps += 1
        best = min(best, loss_value())

    final = loss_value()
    return SSMDistillReceipt(
        schema="NOLANE-L8-SSM-DISTILL-V1",
        pairs=len(pairs),
        optimizer_steps=steps,
        initial_mse=initial,
        final_mse=final,
        best_mse=best,
        mse_improvement=initial-final,
        replacement_parameters=mixer.parameter_count(),
        gradients_seen=gradients_seen,
        config=asdict(config),
    )
