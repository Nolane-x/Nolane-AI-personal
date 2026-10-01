from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StateSpaceCortexConfig:
    latent_dim: int = 32
    state_dim: int = 32
    max_abs_gate: float = 0.20
    initial_gate: float = 0.04

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 < self.max_abs_gate <= 0.30:
            raise ValueError("max_abs_gate must be in (0,0.30]")
        if abs(self.initial_gate) >= self.max_abs_gate:
            raise ValueError("initial_gate must be strictly inside max_abs_gate")


def analytical_state_space_parameter_count(
    hidden_size: int,
    config: StateSpaceCortexConfig | None = None,
) -> int:
    c = config or StateSpaceCortexConfig()
    c.validate()
    h = int(hidden_size)
    d = int(c.state_dim)
    l = int(c.latent_dim)
    if h <= 0:
        raise ValueError("hidden_size must be positive")

    hidden_norm = 2 * h
    latent_norm = l
    hidden_down = h * d + d
    latent_proj = l * d + d
    proposal = d * d + d
    decay = d * d + d
    output_gate = d * d + d
    state_out = d * h + h
    raw_gate = 1
    return (
        hidden_norm
        + latent_norm
        + hidden_down
        + latent_proj
        + proposal
        + decay
        + output_gate
        + state_out
        + raw_gate
    )


class SelectiveStateSpaceCortex:
    """Small selective recurrent state-space path over token sequence.

    This module is intentionally non-attentional. Each token updates a compact
    state using token-conditioned decay/proposal gates, then emits a bounded
    residual back into Qwen hidden space. The same state can persist across
    cached autoregressive generation calls.
    """

    def __init__(
        self,
        hidden_size: int,
        config: StateSpaceCortexConfig | None = None,
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
        self.config = config or StateSpaceCortexConfig()
        self.config.validate()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        c = self.config
        self_outer = self
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))

            class LatentRMSNorm(nn.Module):
                def __init__(self, dim: int, eps: float = 1e-6) -> None:
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(dim))
                    self.eps = float(eps)

                def forward(self, x):
                    return x * torch.rsqrt(
                        x.pow(2).mean(dim=-1, keepdim=True) + self.eps
                    ) * self.weight

            class Module(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.hidden_norm = nn.LayerNorm(self_outer.hidden_size)
                    self.latent_norm = LatentRMSNorm(c.latent_dim)
                    self.hidden_down = nn.Linear(self_outer.hidden_size, c.state_dim)
                    self.latent_proj = nn.Linear(c.latent_dim, c.state_dim)
                    self.proposal = nn.Linear(c.state_dim, c.state_dim)
                    self.decay = nn.Linear(c.state_dim, c.state_dim)
                    self.output_gate = nn.Linear(c.state_dim, c.state_dim)
                    self.state_out = nn.Linear(c.state_dim, self_outer.hidden_size)
                    ratio = c.initial_gate / c.max_abs_gate
                    bias = 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))
                    self.raw_gate = nn.Parameter(torch.tensor(float(bias)))

                    nn.init.zeros_(self.state_out.bias)

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

    def effective_gate(self):
        return self.config.max_abs_gate * self.torch.tanh(self.module.raw_gate)

    def _latent_feature(self, latent, *, device, batch: int):
        torch = self.torch
        if not torch.is_tensor(latent):
            latent = torch.tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("state-space latent shape mismatch")
        latent = latent.to(
            device=device,
            dtype=self.module.latent_proj.weight.dtype,
        )
        if latent.shape[0] == 1 and batch > 1:
            latent = latent.expand(batch, -1)
        if latent.shape[0] != batch:
            raise ValueError("state-space latent batch mismatch")
        latent = self.module.latent_norm(latent)
        return self.torch.tanh(self.module.latent_proj(latent))

    def initial_state(self, latent, *, device, batch: int):
        return self._latent_feature(latent, device=device, batch=batch)

    def scan(self, hidden, latent, *, state=None):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch, sequence, hidden_size]")
        batch, sequence, _ = hidden.shape
        feature = torch.tanh(
            self.module.hidden_down(
                self.module.hidden_norm(hidden).to(dtype=self.module.hidden_down.weight.dtype)
            )
        )
        latent_feature = self._latent_feature(latent, device=hidden.device, batch=batch)
        if state is None:
            state = latent_feature
        else:
            state = state.to(
                device=hidden.device,
                dtype=self.module.hidden_down.weight.dtype,
            )
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if state.shape[0] == 1 and batch > 1:
                state = state.expand(batch, -1)
            if state.shape != (batch, self.config.state_dim):
                raise ValueError("state-space recurrent state shape mismatch")

        outputs = []
        mean_decays = []
        mean_output_gates = []
        for token_index in range(sequence):
            token = feature[:, token_index, :]
            proposal = torch.tanh(
                self.module.proposal(token) + latent_feature
            )
            decay = torch.sigmoid(self.module.decay(token))
            output_gate = torch.sigmoid(self.module.output_gate(token))
            state = decay * state + (1.0 - decay) * proposal
            projected = self.module.state_out(state * output_gate)
            outputs.append(projected)
            mean_decays.append(decay.detach().float().mean())
            mean_output_gates.append(output_gate.detach().float().mean())

        residual = torch.stack(outputs, dim=1).to(dtype=hidden.dtype)
        gate = self.effective_gate().to(device=hidden.device, dtype=hidden.dtype)
        updated = hidden + gate * residual
        trace = {
            "tokens_scanned": int(sequence),
            "effective_gate": float(gate.detach().float().cpu()),
            "mean_decay": float(torch.stack(mean_decays).mean().cpu()),
            "mean_output_gate": float(torch.stack(mean_output_gates).mean().cpu()),
        }
        return updated, state, trace


def state_space_audit(hidden_size: int, config: StateSpaceCortexConfig | None = None) -> dict[str, Any]:
    cortex = SelectiveStateSpaceCortex(hidden_size, config)
    actual = cortex.parameter_count()
    analytical = analytical_state_space_parameter_count(hidden_size, cortex.config)
    if actual != analytical:
        raise RuntimeError(
            f"state-space parameter audit mismatch: actual={actual} analytical={analytical}"
        )
    return {
        "schema": "NOLANE-L12-STATE-SPACE-CORTEX-AUDIT-V1",
        "hidden_size": int(hidden_size),
        "config": asdict(cortex.config),
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "effective_gate": float(cortex.effective_gate().detach().cpu()),
        "status": "DEVELOPMENT_UNPROMOTED",
    }
