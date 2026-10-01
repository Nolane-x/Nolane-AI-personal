from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MultiTimescaleCortexConfig:
    latent_dim: int = 32
    state_dim: int = 24
    max_abs_gate: float = 0.20
    initial_gate: float = 0.04
    slow_decay_floor: float = 0.50

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 < self.max_abs_gate <= 0.30:
            raise ValueError("max_abs_gate must be in (0,0.30]")
        if abs(self.initial_gate) >= self.max_abs_gate:
            raise ValueError("initial_gate must be strictly inside max_abs_gate")
        if not 0.0 <= self.slow_decay_floor < 1.0:
            raise ValueError("slow_decay_floor must be in [0,1)")


def analytical_multiscale_parameter_count(
    hidden_size: int,
    config: MultiTimescaleCortexConfig | None = None,
) -> int:
    c = config or MultiTimescaleCortexConfig()
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
    six_state_linears = 6 * (d * d + d)
    state_out = (2 * d) * h + h
    raw_gate = 1
    return (
        hidden_norm
        + latent_norm
        + hidden_down
        + latent_proj
        + six_state_linears
        + state_out
        + raw_gate
    )


class MultiTimescaleStateSpaceCortex:
    """Fast/slow non-attentional token recurrence for a thin-Qwen scaffold."""

    def __init__(
        self,
        hidden_size: int,
        config: MultiTimescaleCortexConfig | None = None,
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
        self.config = config or MultiTimescaleCortexConfig()
        self.config.validate()
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        c = self.config
        outer = self
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
                    self.hidden_norm = nn.LayerNorm(outer.hidden_size)
                    self.latent_norm = LatentRMSNorm(c.latent_dim)
                    self.hidden_down = nn.Linear(outer.hidden_size, c.state_dim)
                    self.latent_proj = nn.Linear(c.latent_dim, c.state_dim)
                    self.fast_proposal = nn.Linear(c.state_dim, c.state_dim)
                    self.fast_decay = nn.Linear(c.state_dim, c.state_dim)
                    self.slow_proposal = nn.Linear(c.state_dim, c.state_dim)
                    self.slow_decay = nn.Linear(c.state_dim, c.state_dim)
                    self.fast_output_gate = nn.Linear(c.state_dim, c.state_dim)
                    self.slow_output_gate = nn.Linear(c.state_dim, c.state_dim)
                    self.state_out = nn.Linear(2 * c.state_dim, outer.hidden_size)
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
            raise ValueError("multiscale latent shape mismatch")
        latent = latent.to(device=device, dtype=self.module.latent_proj.weight.dtype)
        if latent.shape[0] == 1 and batch > 1:
            latent = latent.expand(batch, -1)
        if latent.shape[0] != batch:
            raise ValueError("multiscale latent batch mismatch")
        latent = self.module.latent_norm(latent)
        return torch.tanh(self.module.latent_proj(latent))

    def initial_state(self, latent, *, device, batch: int):
        feature = self._latent_feature(latent, device=device, batch=batch)
        return self.torch.cat([feature, feature], dim=-1)

    def _unpack_state(self, state, *, latent_feature, batch: int, device):
        d = self.config.state_dim
        if state is None:
            return latent_feature, latent_feature
        state = state.to(device=device, dtype=self.module.hidden_down.weight.dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] == 1 and batch > 1:
            state = state.expand(batch, -1)
        if state.shape != (batch, 2 * d):
            raise ValueError("multiscale recurrent state shape mismatch")
        return state[:, :d], state[:, d:]

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
        fast, slow = self._unpack_state(
            state,
            latent_feature=latent_feature,
            batch=batch,
            device=hidden.device,
        )

        outputs = []
        fast_decays = []
        slow_decays = []
        for token_index in range(sequence):
            token = feature[:, token_index, :]
            fast_proposal = torch.tanh(
                self.module.fast_proposal(token) + latent_feature
            )
            fast_decay = torch.sigmoid(self.module.fast_decay(token))
            fast = fast_decay * fast + (1.0 - fast_decay) * fast_proposal

            slow_proposal = torch.tanh(
                self.module.slow_proposal(fast) + latent_feature
            )
            raw_slow_decay = torch.sigmoid(self.module.slow_decay(token))
            slow_decay = (
                self.config.slow_decay_floor
                + (1.0 - self.config.slow_decay_floor) * raw_slow_decay
            )
            slow = slow_decay * slow + (1.0 - slow_decay) * slow_proposal

            fast_gate = torch.sigmoid(self.module.fast_output_gate(token))
            slow_gate = torch.sigmoid(self.module.slow_output_gate(token))
            joined = torch.cat([fast * fast_gate, slow * slow_gate], dim=-1)
            outputs.append(self.module.state_out(joined))
            fast_decays.append(fast_decay.detach().float().mean())
            slow_decays.append(slow_decay.detach().float().mean())

        residual = torch.stack(outputs, dim=1).to(dtype=hidden.dtype)
        gate = self.effective_gate().to(device=hidden.device, dtype=hidden.dtype)
        updated = hidden + gate * residual
        packed = torch.cat([fast, slow], dim=-1)
        trace = {
            "tokens_scanned": int(sequence),
            "effective_gate": float(gate.detach().float().cpu()),
            "mean_fast_decay": float(torch.stack(fast_decays).mean().cpu()),
            "mean_slow_decay": float(torch.stack(slow_decays).mean().cpu()),
            "slow_decay_floor": float(self.config.slow_decay_floor),
        }
        return updated, packed, trace


def multiscale_cortex_audit(
    hidden_size: int,
    config: MultiTimescaleCortexConfig | None = None,
) -> dict[str, Any]:
    cortex = MultiTimescaleStateSpaceCortex(hidden_size, config)
    actual = cortex.parameter_count()
    analytical = analytical_multiscale_parameter_count(hidden_size, cortex.config)
    if actual != analytical:
        raise RuntimeError(
            f"multiscale parameter audit mismatch: actual={actual} analytical={analytical}"
        )
    return {
        "schema": "NOLANE-L13-MULTISCALE-CORTEX-AUDIT-V1",
        "hidden_size": int(hidden_size),
        "config": asdict(cortex.config),
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "effective_gate": float(cortex.effective_gate().detach().cpu()),
        "status": "DEVELOPMENT_UNPROMOTED",
    }
