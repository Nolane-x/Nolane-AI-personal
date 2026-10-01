from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DeepRecurrentCortexConfig:
    latent_dim: int = 32
    state_dim: int = 20
    virtual_steps: int = 8
    max_virtual_steps: int = 16
    max_abs_gate: float = 0.20
    initial_gate: float = 0.04
    slow_decay_floor: float = 0.55

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.virtual_steps < 1:
            raise ValueError("virtual_steps must be >=1")
        if self.max_virtual_steps < self.virtual_steps:
            raise ValueError("max_virtual_steps must be >= virtual_steps")
        if not 0.0 < self.max_abs_gate <= 0.30:
            raise ValueError("max_abs_gate must be in (0,0.30]")
        if abs(self.initial_gate) >= self.max_abs_gate:
            raise ValueError("initial_gate must be strictly inside max_abs_gate")
        if not 0.0 <= self.slow_decay_floor < 1.0:
            raise ValueError("slow_decay_floor must be in [0,1)")


def analytical_deep_recurrent_parameter_count(
    hidden_size: int,
    config: DeepRecurrentCortexConfig | None = None,
) -> int:
    c = config or DeepRecurrentCortexConfig()
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
    temporal_linears = 6 * (d * d + d)
    virtual_depth_linears = 3 * (d * d + d)
    virtual_step_embedding = c.max_virtual_steps * d
    state_out = (3 * d) * h + h
    raw_gate = 1
    return (
        hidden_norm
        + latent_norm
        + hidden_down
        + latent_proj
        + temporal_linears
        + virtual_depth_linears
        + virtual_step_embedding
        + state_out
        + raw_gate
    )


class DeepRecurrentStateSpaceCortex:
    """Fast/slow temporal recurrence plus shared virtual-depth micro-steps.

    The temporal states carry information across tokens. The depth state then
    iterates through a small number of shared recurrent micro-steps for each
    token, approximating the composition of many removed Transformer blocks
    without introducing parameters proportional to removed Qwen depth.
    """

    def __init__(
        self,
        hidden_size: int,
        config: DeepRecurrentCortexConfig | None = None,
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
        self.config = config or DeepRecurrentCortexConfig()
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

                    self.depth_transition = nn.Linear(c.state_dim, c.state_dim)
                    self.depth_token = nn.Linear(c.state_dim, c.state_dim)
                    self.depth_gate = nn.Linear(c.state_dim, c.state_dim)
                    self.depth_embedding = nn.Parameter(
                        torch.empty(c.max_virtual_steps, c.state_dim)
                    )
                    nn.init.normal_(self.depth_embedding, mean=0.0, std=0.02)

                    self.state_out = nn.Linear(3 * c.state_dim, outer.hidden_size)
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
            raise ValueError("deep-recurrent latent shape mismatch")
        latent = latent.to(device=device, dtype=self.module.latent_proj.weight.dtype)
        if latent.shape[0] == 1 and batch > 1:
            latent = latent.expand(batch, -1)
        if latent.shape[0] != batch:
            raise ValueError("deep-recurrent latent batch mismatch")
        latent = self.module.latent_norm(latent)
        return self.torch.tanh(self.module.latent_proj(latent))

    def initial_state(self, latent, *, device, batch: int):
        feature = self._latent_feature(latent, device=device, batch=batch)
        return self.torch.cat([feature, feature, feature], dim=-1)

    def _unpack_state(self, state, *, latent_feature, batch: int, device):
        d = self.config.state_dim
        if state is None:
            return latent_feature, latent_feature, latent_feature
        state = state.to(device=device, dtype=self.module.hidden_down.weight.dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] == 1 and batch > 1:
            state = state.expand(batch, -1)
        if state.shape != (batch, 3 * d):
            raise ValueError("deep-recurrent packed state shape mismatch")
        return state[:, :d], state[:, d:2*d], state[:, 2*d:]

    def scan(self, hidden, latent, *, state=None, virtual_steps: int | None = None):
        torch = self.torch
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("expected hidden [batch, sequence, hidden_size]")
        steps = self.config.virtual_steps if virtual_steps is None else int(virtual_steps)
        if not 1 <= steps <= self.config.max_virtual_steps:
            raise ValueError("virtual_steps override out of range")

        batch, sequence, _ = hidden.shape
        feature = torch.tanh(
            self.module.hidden_down(
                self.module.hidden_norm(hidden).to(dtype=self.module.hidden_down.weight.dtype)
            )
        )
        latent_feature = self._latent_feature(latent, device=hidden.device, batch=batch)
        fast, slow, depth = self._unpack_state(
            state,
            latent_feature=latent_feature,
            batch=batch,
            device=hidden.device,
        )

        outputs = []
        fast_decays = []
        slow_decays = []
        depth_carries = []
        for token_index in range(sequence):
            token = feature[:, token_index, :]

            fast_proposal = torch.tanh(self.module.fast_proposal(token) + latent_feature)
            fast_decay = torch.sigmoid(self.module.fast_decay(token))
            fast = fast_decay * fast + (1.0 - fast_decay) * fast_proposal

            slow_proposal = torch.tanh(self.module.slow_proposal(fast) + latent_feature)
            raw_slow_decay = torch.sigmoid(self.module.slow_decay(token))
            slow_decay = (
                self.config.slow_decay_floor
                + (1.0 - self.config.slow_decay_floor) * raw_slow_decay
            )
            slow = slow_decay * slow + (1.0 - slow_decay) * slow_proposal

            depth = 0.5 * depth + 0.25 * fast + 0.25 * slow
            for step in range(steps):
                step_feature = self.module.depth_embedding[step].unsqueeze(0)
                proposal = torch.tanh(
                    self.module.depth_transition(depth)
                    + self.module.depth_token(token)
                    + latent_feature
                    + step_feature
                )
                carry = torch.sigmoid(self.module.depth_gate(depth + step_feature))
                depth = carry * depth + (1.0 - carry) * proposal
                depth_carries.append(carry.detach().float().mean())

            fast_gate = torch.sigmoid(self.module.fast_output_gate(token))
            slow_gate = torch.sigmoid(self.module.slow_output_gate(token))
            joined = torch.cat([fast * fast_gate, slow * slow_gate, depth], dim=-1)
            outputs.append(self.module.state_out(joined))
            fast_decays.append(fast_decay.detach().float().mean())
            slow_decays.append(slow_decay.detach().float().mean())

        residual = torch.stack(outputs, dim=1).to(dtype=hidden.dtype)
        gate = self.effective_gate().to(device=hidden.device, dtype=hidden.dtype)
        updated = hidden + gate * residual
        packed = torch.cat([fast, slow, depth], dim=-1)
        trace = {
            "tokens_scanned": int(sequence),
            "virtual_steps": int(steps),
            "virtual_microsteps": int(sequence * steps),
            "effective_gate": float(gate.detach().float().cpu()),
            "mean_fast_decay": float(torch.stack(fast_decays).mean().cpu()),
            "mean_slow_decay": float(torch.stack(slow_decays).mean().cpu()),
            "mean_depth_carry": float(torch.stack(depth_carries).mean().cpu()),
            "slow_decay_floor": float(self.config.slow_decay_floor),
        }
        return updated, packed, trace


def deep_recurrent_cortex_audit(
    hidden_size: int,
    config: DeepRecurrentCortexConfig | None = None,
) -> dict[str, Any]:
    cortex = DeepRecurrentStateSpaceCortex(hidden_size, config)
    actual = cortex.parameter_count()
    analytical = analytical_deep_recurrent_parameter_count(hidden_size, cortex.config)
    if actual != analytical:
        raise RuntimeError(
            f"deep-recurrent parameter audit mismatch: actual={actual} analytical={analytical}"
        )
    return {
        "schema": "NOLANE-L14-DEEP-RECURRENT-CORTEX-AUDIT-V1",
        "hidden_size": int(hidden_size),
        "config": asdict(cortex.config),
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "effective_gate": float(cortex.effective_gate().detach().cpu()),
        "status": "DEVELOPMENT_UNPROMOTED",
    }
