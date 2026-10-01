from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .surgery import resolve_transformer_layers, select_layer_indices


@dataclass(slots=True)
class LivingBridgeConfig:
    latent_dim: int = 32
    bridge_dim: int = 32
    max_layers: int = 64
    max_abs_gate: float = 0.15
    initial_gate: float = 0.02

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.bridge_dim <= 0:
            raise ValueError("bridge_dim must be positive")
        if self.max_layers <= 0:
            raise ValueError("max_layers must be positive")
        if not 0.0 < self.max_abs_gate <= 0.25:
            raise ValueError("max_abs_gate must be in (0, 0.25]")
        if abs(self.initial_gate) >= self.max_abs_gate:
            raise ValueError("initial_gate must be strictly inside max_abs_gate")


def analytical_bridge_parameter_count(
    hidden_size: int,
    config: LivingBridgeConfig | None = None,
) -> int:
    c = config or LivingBridgeConfig()
    c.validate()
    h = int(hidden_size)
    b = int(c.bridge_dim)
    l = int(c.latent_dim)
    if h <= 0:
        raise ValueError("hidden_size must be positive")

    latent_norm = l
    hidden_norm = 2 * h
    hidden_down = h * b + b
    latent_proj = l * b + b
    layer_embedding = c.max_layers * b
    gru_input = 3 * b
    gru = 3 * b * gru_input + 3 * b * b + 6 * b
    residual_up = b * h + h
    gate_head = b + 1
    return (
        latent_norm
        + hidden_norm
        + hidden_down
        + latent_proj
        + layer_embedding
        + gru
        + residual_up
        + gate_head
    )


class CrossLayerLivingBridge:
    """A recurrent non-Transformer path threaded through selected decoder blocks.

    The persistent 32D Living latent is fused with the current decoder hidden
    summary and a learned layer identity. A GRU state then recurs *across model
    depth* and emits both a residual and a bounded gate for each selected Qwen
    block. Base-Qwen parameters are never part of this module.
    """

    def __init__(
        self,
        hidden_size: int,
        config: LivingBridgeConfig | None = None,
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
        self.config = config or LivingBridgeConfig()
        self.config.validate()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))

            class Module(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    class LatentRMSNorm(nn.Module):
                        def __init__(self, dim: int, eps: float = 1e-6) -> None:
                            super().__init__()
                            self.weight = nn.Parameter(torch.ones(dim))
                            self.eps = float(eps)

                        def forward(self, x):
                            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                            return x * rms * self.weight

                    self.latent_norm = LatentRMSNorm(c.latent_dim)
                    self.hidden_norm = nn.LayerNorm(self_outer.hidden_size)
                    self.hidden_down = nn.Linear(self_outer.hidden_size, c.bridge_dim)
                    self.latent_proj = nn.Linear(c.latent_dim, c.bridge_dim)
                    self.layer_embedding = nn.Embedding(c.max_layers, c.bridge_dim)
                    self.recurrent = nn.GRUCell(3 * c.bridge_dim, c.bridge_dim)
                    self.residual_up = nn.Linear(c.bridge_dim, self_outer.hidden_size)
                    self.gate_head = nn.Linear(c.bridge_dim, 1)

                    nn.init.zeros_(self.gate_head.weight)
                    ratio = c.initial_gate / c.max_abs_gate
                    bias = 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))
                    nn.init.constant_(self.gate_head.bias, bias)

            self_outer = self
            self.module = Module()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def to(self, device: str):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    def _latent_tensor(self, latent, *, device):
        torch = self.torch
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent shape mismatch")
        return latent.to(device=device, dtype=self.module.latent_proj.weight.dtype)

    def latent_feature(self, latent, *, device):
        torch = self.torch
        latent = self._latent_tensor(latent, device=device)
        return torch.tanh(self.module.latent_proj(self.module.latent_norm(latent)))

    def initial_state(self, latent, *, device):
        return self.latent_feature(latent, device=device)

    def step(self, hidden, latent_feature, state, *, layer_index: int):
        torch = self.torch
        if hidden.ndim != 3:
            raise ValueError("expected hidden state shaped [batch, sequence, hidden]")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("bridge hidden-size mismatch")
        if not 0 <= int(layer_index) < self.config.max_layers:
            raise ValueError("layer index exceeds bridge max_layers")

        summary = hidden.float().mean(dim=1).to(dtype=self.module.hidden_down.weight.dtype)
        hidden_feature = torch.tanh(
            self.module.hidden_down(self.module.hidden_norm(summary))
        )

        batch = hidden.shape[0]
        if latent_feature.shape[0] == 1 and batch > 1:
            latent_feature = latent_feature.expand(batch, -1)
        if state.shape[0] == 1 and batch > 1:
            state = state.expand(batch, -1)
        if latent_feature.shape[0] != batch or state.shape[0] != batch:
            raise ValueError("bridge batch mismatch")

        layer_ids = torch.full(
            (batch,),
            int(layer_index),
            dtype=torch.long,
            device=hidden.device,
        )
        layer_feature = self.module.layer_embedding(layer_ids)
        recurrent_input = torch.cat(
            [hidden_feature, latent_feature, layer_feature],
            dim=-1,
        )
        next_state = self.module.recurrent(recurrent_input, state)
        residual = self.module.residual_up(torch.tanh(next_state)).to(dtype=hidden.dtype)
        gate = self.config.max_abs_gate * torch.tanh(self.module.gate_head(next_state))
        return next_state, residual, gate

    def effective_gate_bounds(self) -> tuple[float, float]:
        return (-self.config.max_abs_gate, self.config.max_abs_gate)


def _inject(hidden, residual, gate, *, token_scope: str):
    if residual.ndim != 2 or gate.ndim != 2:
        raise ValueError("bridge residual/gate shape mismatch")
    delta = (residual * gate).unsqueeze(1)
    if token_scope == "all":
        return hidden + delta
    if token_scope != "last":
        raise ValueError("token_scope must be 'last' or 'all'")
    updated = hidden.clone()
    updated[:, -1:, :] = updated[:, -1:, :] + delta
    return updated


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden


class LivingBridgeHookSession(contextlib.AbstractContextManager):
    """Installs a depth-recurrent bridge for one model forward/generation scope.

    The ephemeral bridge state resets at the first selected decoder layer on
    every model forward. During a forward it recurs through selected layers.
    This avoids accidentally carrying autograd state between generation steps.
    """

    def __init__(
        self,
        model,
        bridge: CrossLayerLivingBridge,
        latent,
        *,
        layer_indices: Iterable[int] | None = None,
        token_scope: str = "all",
    ) -> None:
        self.model = model
        self.bridge = bridge
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        self.layer_indices = select_layer_indices(len(self.layers), layer_indices)
        if max(self.layer_indices) >= bridge.config.max_layers:
            raise ValueError("selected layer exceeds bridge max_layers")
        if token_scope not in {"last", "all"}:
            raise ValueError("token_scope must be 'last' or 'all'")
        self.token_scope = token_scope
        self.first_layer = self.layer_indices[0]
        self.handles: list[Any] = []
        self.state = None
        self.latent_feature = None
        self.trace: list[dict[str, float | int]] = []

    def __enter__(self):
        self.trace.clear()
        self.state = None
        self.latent_feature = None

        for index in self.layer_indices:
            layer = self.layers[index]

            def hook(_module, _inputs, output, *, _index=index):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if _index == self.first_layer:
                    self.latent_feature = self.bridge.latent_feature(
                        self.latent,
                        device=hidden.device,
                    )
                    self.state = self.latent_feature
                if self.state is None or self.latent_feature is None:
                    raise RuntimeError("living bridge state was not initialized")

                self.state, residual, gate = self.bridge.step(
                    hidden,
                    self.latent_feature,
                    self.state,
                    layer_index=_index,
                )
                gate_value = float(gate.detach().float().mean().cpu())
                residual_norm = float(
                    residual.detach().float().norm(dim=-1).mean().cpu()
                )
                self.trace.append(
                    {
                        "layer_index": int(_index),
                        "gate_mean": gate_value,
                        "residual_norm": residual_norm,
                    }
                )
                replaced = _inject(
                    hidden,
                    residual,
                    gate.to(dtype=residual.dtype),
                    token_scope=self.token_scope,
                )
                return _replace_hidden(output, replaced)

            self.handles.append(layer.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.state = None
        self.latent_feature = None
        return False


def bridge_audit(hidden_size: int, config: LivingBridgeConfig | None = None) -> dict[str, Any]:
    bridge = CrossLayerLivingBridge(hidden_size, config)
    actual = bridge.parameter_count()
    analytical = analytical_bridge_parameter_count(hidden_size, bridge.config)
    if actual != analytical:
        raise RuntimeError(
            f"bridge parameter audit mismatch: actual={actual} analytical={analytical}"
        )
    low, high = bridge.effective_gate_bounds()
    return {
        "schema": "NOLANE-L7-LIVING-BRIDGE-AUDIT-V1",
        "hidden_size": int(hidden_size),
        "config": asdict(bridge.config),
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "gate_bounds": [low, high],
        "status": "DEVELOPMENT_UNPROMOTED",
    }
