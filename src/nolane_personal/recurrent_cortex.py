from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterable

from .surgery import parameter_guard_snapshot, resolve_transformer_layers, select_layer_indices


@dataclass(slots=True)
class RecurrentCortexConfig:
    latent_dim: int = 32
    recurrent_dim: int = 24
    max_abs_gate: float = 0.10

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.recurrent_dim <= 0:
            raise ValueError("recurrent_dim must be positive")
        if not 0.0 < self.max_abs_gate <= 0.25:
            raise ValueError("max_abs_gate must be in (0, 0.25]")


def analytical_recurrent_parameter_count(
    hidden_size: int,
    config: RecurrentCortexConfig | None = None,
) -> int:
    c = config or RecurrentCortexConfig()
    c.validate()
    h = int(hidden_size)
    r = int(c.recurrent_dim)
    l = int(c.latent_dim)
    if h <= 0:
        raise ValueError("hidden_size must be positive")

    hidden_norm = 2 * h
    input_proj = h * r + r
    latent_proj = l * r + r
    gru_cell = 6 * r * r + 6 * r
    output_proj = r * h + h
    gate = 1
    return hidden_norm + input_proj + latent_proj + gru_cell + output_proj + gate


class RecurrentCortexMixer:
    """Shared recurrent neural mixer inserted after selected Qwen decoder layers.

    The mixer scans token representations in sequence order and carries a
    recurrent state. Selected decoder layers share the mixer weights but keep
    independent recurrent states.
    """

    def __init__(
        self,
        hidden_size: int,
        config: RecurrentCortexConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        self.torch = torch
        self.nn = nn
        self.hidden_size = int(hidden_size)
        self.config = config or RecurrentCortexConfig()
        self.config.validate()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.module = nn.ModuleDict(
                {
                    "hidden_norm": nn.LayerNorm(self.hidden_size),
                    "input_proj": nn.Linear(self.hidden_size, c.recurrent_dim),
                    "latent_proj": nn.Linear(c.latent_dim, c.recurrent_dim),
                    "cell": nn.GRUCell(c.recurrent_dim, c.recurrent_dim),
                    "output_proj": nn.Linear(c.recurrent_dim, self.hidden_size),
                }
            )
            self.raw_gate = nn.Parameter(torch.zeros(()))
            self.module.register_parameter("raw_gate", self.raw_gate)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def effective_gate(self, override: float | None = None):
        torch = self.torch
        if override is not None:
            value = max(-self.config.max_abs_gate, min(self.config.max_abs_gate, float(override)))
            return torch.tensor(value, dtype=self.raw_gate.dtype, device=self.raw_gate.device)
        return self.config.max_abs_gate * torch.tanh(self.raw_gate)

    def initial_state(self, latent, *, batch_size: int, device, dtype):
        torch = self.torch
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent shape mismatch")
        if latent.shape[0] == 1 and batch_size > 1:
            latent = latent.expand(batch_size, -1)
        if latent.shape[0] != batch_size:
            raise ValueError("latent batch mismatch")
        weight_dtype = self.module["latent_proj"].weight.dtype
        latent = latent.to(device=device, dtype=weight_dtype)
        state = self.torch.tanh(self.module["latent_proj"](latent))
        return state.to(dtype=dtype)

    def scan(self, hidden, latent, state=None, *, gate_override: float | None = None):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden state [batch, sequence, hidden_size]")
        batch, seq, _ = hidden.shape
        work_dtype = self.module["input_proj"].weight.dtype
        if state is None:
            state = self.initial_state(
                latent,
                batch_size=batch,
                device=hidden.device,
                dtype=work_dtype,
            )
        else:
            if state.ndim != 2 or state.shape != (batch, self.config.recurrent_dim):
                raise ValueError("recurrent state shape mismatch")
            state = state.to(device=hidden.device, dtype=work_dtype)

        gate = self.effective_gate(gate_override).to(device=hidden.device, dtype=work_dtype)
        outputs = []
        for position in range(seq):
            token = hidden[:, position, :].to(dtype=work_dtype)
            token = self.module["hidden_norm"](token)
            projected = torch.nn.functional.silu(self.module["input_proj"](token))
            state = self.module["cell"](projected, state)
            delta = self.module["output_proj"](state) * gate
            outputs.append(hidden[:, position, :] + delta.to(dtype=hidden.dtype))
        return torch.stack(outputs, dim=1), state

    def to(self, device: str):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden


class RecurrentHookSession(contextlib.AbstractContextManager):
    """Attach the shared mixer to selected Qwen layers for one model session."""

    def __init__(
        self,
        model,
        mixer: RecurrentCortexMixer,
        latent,
        *,
        layer_indices: Iterable[int] | None = None,
        initial_states: dict[int, Any] | None = None,
        gate_override: float | None = None,
    ) -> None:
        self.model = model
        self.mixer = mixer
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        self.layer_indices = select_layer_indices(len(self.layers), layer_indices)
        self.states: dict[int, Any] = dict(initial_states or {})
        self.gate_override = gate_override
        self.handles: list[Any] = []

    def __enter__(self):
        for index in self.layer_indices:
            layer = self.layers[index]

            def hook(_module, _inputs, output, *, _index=index):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                mixed, state = self.mixer.scan(
                    hidden,
                    self.latent,
                    self.states.get(_index),
                    gate_override=self.gate_override,
                )
                self.states[_index] = state
                return _replace_hidden(output, mixed)

            self.handles.append(layer.register_forward_hook(hook))
        return self

    def detached_states(self) -> dict[int, Any]:
        return {index: state.detach().clone() for index, state in self.states.items()}

    def cpu_state_values(self) -> dict[int, list[list[float]]]:
        return {
            index: state.detach().float().cpu().tolist()
            for index, state in self.states.items()
        }

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return False


@dataclass(slots=True)
class HybridCortexConfig:
    layer_indices: tuple[int, ...] | None = None
    carry_across_calls: bool = True

    def validate(self) -> None:
        if self.layer_indices is not None and not self.layer_indices:
            raise ValueError("layer_indices cannot be empty")


class HybridRecurrentCortex:
    """Frozen Qwen plus a trainable recurrent neural pathway.

    Qwen remains frozen. The recurrent mixer owns trainable weights and carries
    per-layer recurrent states that can persist across calls.
    """

    def __init__(
        self,
        model,
        mixer: RecurrentCortexMixer,
        latent,
        *,
        config: HybridCortexConfig | None = None,
    ) -> None:
        self.model = model
        self.mixer = mixer
        self.latent = latent
        self.config = config or HybridCortexConfig()
        self.config.validate()
        self.recurrent_states: dict[int, Any] = {}
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_recurrent_state(self) -> None:
        self.recurrent_states = {}

    def trainable_parameters(self):
        return [p for p in self.mixer.module.parameters() if p.requires_grad]

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("recurrent mixer exposes no trainable parameters")

    def forward(self, *, carry_state: bool | None = None, **model_inputs):
        self.assert_gradient_boundary()
        carry = self.config.carry_across_calls if carry_state is None else bool(carry_state)
        initial = self.recurrent_states if carry else {}
        with RecurrentHookSession(
            self.model,
            self.mixer,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=initial,
        ) as session:
            output = self.model(**model_inputs)
            if carry:
                self.recurrent_states = session.detached_states()
            return output

    def generate(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.mixer.eval()
        with RecurrentHookSession(
            self.model,
            self.mixer,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=self.recurrent_states,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.recurrent_states = session.detached_states()
            return output

    def base_guard(self):
        return parameter_guard_snapshot(self.model)
