from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .surgery import resolve_transformer_layers, select_layer_indices


@dataclass(slots=True)
class BlockReplacementConfig:
    latent_dim: int = 32
    recurrent_dim: int = 32
    max_layers: int = 64
    initial_gate: float = 0.05

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.recurrent_dim <= 0:
            raise ValueError("recurrent_dim must be positive")
        if self.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("initial_gate must be in (0,1)")


def analytical_replacement_parameter_count(hidden_size: int, config: BlockReplacementConfig | None = None) -> int:
    c = config or BlockReplacementConfig()
    c.validate()
    h, r, l = int(hidden_size), int(c.recurrent_dim), int(c.latent_dim)
    if h <= 0:
        raise ValueError("hidden_size must be positive")
    latent_norm = l
    hidden_norm = 2 * h
    hidden_down = h * r + r
    latent_proj = l * r + r
    layer_embedding = c.max_layers * r
    gru_input = 3 * r
    gru = 3 * r * gru_input + 3 * r * r + 6 * r
    residual_up = r * h + h
    gate_head = r + 1
    return latent_norm + hidden_norm + hidden_down + latent_proj + layer_embedding + gru + residual_up + gate_head


class RecurrentBlockReplacement:
    """Shared recurrent substitute for selected frozen Transformer decoder blocks.

    In bypass mode the original decoder block is not called. The replacement
    maps the incoming hidden sequence through a compact GRU conditioned on the
    persistent Living latent and layer identity, then predicts a gated residual.
    """

    def __init__(self, hidden_size: int, config: BlockReplacementConfig | None = None, *, seed: int = 0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        self.torch, self.nn = torch, nn
        self.hidden_size = int(hidden_size)
        self.config = config or BlockReplacementConfig()
        self.config.validate()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        c = self.config
        self_outer = self
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))

            class RMSNorm(nn.Module):
                def __init__(self, dim: int, eps: float = 1e-6) -> None:
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(dim))
                    self.eps = float(eps)

                def forward(self, x):
                    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

            class Module(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.latent_norm = RMSNorm(c.latent_dim)
                    self.hidden_norm = nn.LayerNorm(self_outer.hidden_size)
                    self.hidden_down = nn.Linear(self_outer.hidden_size, c.recurrent_dim)
                    self.latent_proj = nn.Linear(c.latent_dim, c.recurrent_dim)
                    self.layer_embedding = nn.Embedding(c.max_layers, c.recurrent_dim)
                    self.recurrent = nn.GRU(3 * c.recurrent_dim, c.recurrent_dim, batch_first=True)
                    self.residual_up = nn.Linear(c.recurrent_dim, self_outer.hidden_size)
                    self.gate_head = nn.Linear(c.recurrent_dim, 1)
                    nn.init.zeros_(self.gate_head.weight)
                    bias = math.log(c.initial_gate / (1.0 - c.initial_gate))
                    nn.init.constant_(self.gate_head.bias, bias)

            self.module = Module()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def to(self, device: str):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    def _latent_feature(self, latent, *, batch: int, device):
        torch = self.torch
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent shape mismatch")
        if latent.shape[0] == 1 and batch > 1:
            latent = latent.expand(batch, -1)
        if latent.shape[0] != batch:
            raise ValueError("latent batch mismatch")
        dtype = self.module.latent_proj.weight.dtype
        latent = latent.to(device=device, dtype=dtype)
        return self.torch.tanh(self.module.latent_proj(self.module.latent_norm(latent)))

    def replace(self, hidden, latent, *, layer_index: int, state=None):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch, sequence, hidden_size]")
        if not 0 <= int(layer_index) < self.config.max_layers:
            raise ValueError("layer index exceeds max_layers")

        batch, seq, _ = hidden.shape
        dtype = self.module.hidden_down.weight.dtype
        normalized = self.module.hidden_norm(hidden.to(dtype=dtype))
        hidden_feature = torch.nn.functional.silu(self.module.hidden_down(normalized))
        latent_feature = self._latent_feature(latent, batch=batch, device=hidden.device)
        latent_seq = latent_feature.unsqueeze(1).expand(batch, seq, -1)
        ids = torch.full((batch, seq), int(layer_index), dtype=torch.long, device=hidden.device)
        layer_feature = self.module.layer_embedding(ids)
        recurrent_input = torch.cat([hidden_feature, latent_seq, layer_feature], dim=-1)

        h0 = None
        if state is not None:
            if state.ndim != 2 or state.shape != (batch, self.config.recurrent_dim):
                raise ValueError("replacement recurrent state shape mismatch")
            h0 = state.to(device=hidden.device, dtype=dtype).unsqueeze(0)

        recurrent_out, next_state = self.module.recurrent(recurrent_input, h0)
        residual = self.module.residual_up(torch.tanh(recurrent_out))
        gate = torch.sigmoid(self.module.gate_head(recurrent_out))
        replaced = hidden + (residual * gate).to(dtype=hidden.dtype)
        return replaced, next_state.squeeze(0), gate

    def audit(self) -> dict[str, Any]:
        actual = self.parameter_count()
        analytical = analytical_replacement_parameter_count(self.hidden_size, self.config)
        if actual != analytical:
            raise RuntimeError(f"replacement parameter mismatch: {actual} != {analytical}")
        return {
            "schema": "NOLANE-L9-RECURRENT-BLOCK-REPLACEMENT-AUDIT-V1",
            "hidden_size": self.hidden_size,
            "config": asdict(self.config),
            "parameter_count": actual,
            "analytical_parameter_count": analytical,
            "status": "DEVELOPMENT_UNPROMOTED",
        }


class BlockReplacementSession(contextlib.AbstractContextManager):
    """Temporarily bypass selected decoder blocks by replacing their forward methods."""

    def __init__(
        self,
        model,
        replacement: RecurrentBlockReplacement,
        latent,
        *,
        layer_indices: Iterable[int] | None = None,
        initial_states: dict[int, Any] | None = None,
        reset_states_each_model_forward: bool = False,
    ) -> None:
        self.model = model
        self.replacement = replacement
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        self.layer_indices = select_layer_indices(len(self.layers), layer_indices)
        if max(self.layer_indices) >= replacement.config.max_layers:
            raise ValueError("selected layer exceeds replacement max_layers")
        self.initial_states = dict(initial_states or {})
        self.states = dict(self.initial_states)
        self.reset_states_each_model_forward = bool(reset_states_each_model_forward)
        self.model_pre_handle = None
        self.model_forward_count = 0
        self.original_forwards: dict[int, Any] = {}
        self.bypass_counts: dict[int, int] = {i: 0 for i in self.layer_indices}
        self.gate_means: dict[int, list[float]] = {i: [] for i in self.layer_indices}

    def __enter__(self):
        def model_forward_hook(_module, _args, _kwargs):
            self.model_forward_count += 1
            if self.reset_states_each_model_forward:
                self.states = {
                    index: state.detach().clone()
                    for index, state in self.initial_states.items()
                }

        self.model_pre_handle = self.model.register_forward_pre_hook(
            model_forward_hook,
            with_kwargs=True,
        )

        for index in self.layer_indices:
            layer = self.layers[index]
            self.original_forwards[index] = layer.forward

            def replacement_forward(*args, _index=index, **kwargs):
                if args:
                    hidden = args[0]
                else:
                    hidden = kwargs.get("hidden_states")
                if hidden is None:
                    raise ValueError("decoder layer hidden_states missing")
                replaced, state, gate = self.replacement.replace(
                    hidden,
                    self.latent,
                    layer_index=_index,
                    state=self.states.get(_index),
                )
                self.states[_index] = state
                self.bypass_counts[_index] += 1
                self.gate_means[_index].append(float(gate.detach().float().mean().cpu()))
                return replaced

            layer.forward = replacement_forward
        return self

    def detached_states(self) -> dict[int, Any]:
        return {i: state.detach().clone() for i, state in self.states.items()}

    def __exit__(self, exc_type, exc, tb):
        for index in self.layer_indices:
            self.layers[index].forward = self.original_forwards[index]
        self.original_forwards.clear()
        if self.model_pre_handle is not None:
            self.model_pre_handle.remove()
            self.model_pre_handle = None
        return False
