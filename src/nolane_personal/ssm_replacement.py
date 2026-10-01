from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from .surgery import module_parameter_digest


@dataclass(slots=True)
class SSMReplacementConfig:
    latent_dim: int = 32
    state_dim: int = 24
    max_abs_gate: float = 1.0

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 < self.max_abs_gate <= 1.0:
            raise ValueError("max_abs_gate must be in (0,1]")


def analytical_ssm_parameter_count(
    hidden_size: int,
    config: SSMReplacementConfig | None = None,
) -> int:
    c = config or SSMReplacementConfig()
    c.validate()
    h = int(hidden_size)
    d = int(c.state_dim)
    l = int(c.latent_dim)
    if h <= 0:
        raise ValueError("hidden_size must be positive")
    hidden_norm = 2 * h
    input_proj = h * d + d
    latent_proj = l * d + d
    decay = d
    output_proj = d * h + h
    gate = 1
    return hidden_norm + input_proj + latent_proj + decay + output_proj + gate


class SelectiveStateSpaceMixer:
    """Small diagonal state-space replacement for one Transformer block.

    s[t+1] = a * s[t] + (1-a) * phi(W_in LN(h[t]))
    h'[t]  = h[t] + gate * W_out(s[t+1])

    The initial state is conditioned on the persistent Living latent.
    """

    def __init__(
        self,
        hidden_size: int,
        config: SSMReplacementConfig | None = None,
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
        self.config = config or SSMReplacementConfig()
        self.config.validate()
        c = self.config

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.module = nn.ModuleDict(
                {
                    "hidden_norm": nn.LayerNorm(self.hidden_size),
                    "input_proj": nn.Linear(self.hidden_size, c.state_dim),
                    "latent_proj": nn.Linear(c.latent_dim, c.state_dim),
                    "output_proj": nn.Linear(c.state_dim, self.hidden_size),
                }
            )
            self.decay_logits = nn.Parameter(torch.zeros(c.state_dim))
            self.raw_gate = nn.Parameter(torch.zeros(()))
            self.module.register_parameter("decay_logits", self.decay_logits)
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
        return self.torch.tanh(self.module["latent_proj"](latent)).to(dtype=dtype)

    def scan(self, hidden, latent, state=None, *, gate_override: float | None = None):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch, sequence, hidden_size]")
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
            state = state.to(device=hidden.device, dtype=work_dtype)
            if state.shape != (batch, self.config.state_dim):
                raise ValueError("state shape mismatch")

        decay = torch.sigmoid(self.decay_logits).to(device=hidden.device, dtype=work_dtype)
        gate = self.effective_gate(gate_override).to(device=hidden.device, dtype=work_dtype)
        outputs = []
        for position in range(seq):
            x = hidden[:, position, :].to(dtype=work_dtype)
            x = self.module["hidden_norm"](x)
            drive = torch.nn.functional.silu(self.module["input_proj"](x))
            state = decay * state + (1.0 - decay) * drive
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


class StateSpaceReplacementBlock:
    """nn.Module-compatible Qwen decoder-layer replacement."""

    def __init__(self, mixer: SelectiveStateSpaceMixer, latent) -> None:
        nn = mixer.nn

        class _Block(nn.Module):
            def __init__(self, outer) -> None:
                super().__init__()
                self.outer = outer
                self.mixer_module = outer.mixer.module

            def forward(self, hidden_states, *args, **kwargs):
                del args, kwargs
                mixed, state = self.outer.mixer.scan(
                    hidden_states,
                    self.outer.latent,
                    self.outer.state,
                )
                self.outer.state = state
                self.outer.calls += 1
                return mixed

        self.mixer = mixer
        self.latent = latent
        self.state = None
        self.calls = 0
        self.module = _Block(self)

    def reset_state(self) -> None:
        self.state = None

    def digest(self) -> str:
        return module_parameter_digest(self.mixer.module)


def resolve_layer_container(model):
    paths = ("model.layers", "model.model.layers", "transformer.h", "layers")
    for path in paths:
        current = model
        ok = True
        for part in path.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok and hasattr(current, "__getitem__") and hasattr(current, "__setitem__"):
            return current
    raise ValueError("unable to resolve mutable decoder-layer container")


class BlockReplacementSession(contextlib.AbstractContextManager):
    """Temporarily replace one actual Transformer decoder layer.

    The original layer is not called while the session is active.
    """

    def __init__(self, model, layer_index: int, replacement: StateSpaceReplacementBlock) -> None:
        self.model = model
        self.container = resolve_layer_container(model)
        self.layer_index = int(layer_index)
        if self.layer_index < 0 or self.layer_index >= len(self.container):
            raise ValueError("replacement layer index out of range")
        self.replacement = replacement
        self.original = None

    def __enter__(self):
        self.original = self.container[self.layer_index]
        self.container[self.layer_index] = self.replacement.module
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.container[self.layer_index] = self.original
            self.original = None
        return False
