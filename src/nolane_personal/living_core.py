from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .events import LivingEvent
from .state import LivingState


OBSERVED_STATE_DIM = 15


def state_vector(state: LivingState) -> list[float]:
    unresolved = sum(1 for t in state.open_threads if t.unresolved)
    return [
        state.affect.valence,
        state.affect.arousal,
        state.affect.energy,
        state.affect.playfulness,
        state.affect.irritation,
        state.affect.concern,
        state.affect.social_drive,
        state.relationship.closeness,
        state.relationship.trust,
        state.relationship.familiarity,
        state.working.curiosity,
        state.working.uncertainty,
        min(1.0, unresolved / 8.0),
        1.0 if state.rest_mode else 0.0,
        min(1.0, state.relationship.interaction_count / 1000.0),
    ]


@dataclass(slots=True)
class LivingCoreConfig:
    event_dim: int = 48
    observed_state_dim: int = OBSERVED_STATE_DIM
    latent_dim: int = 32
    hidden_dim: int = 64
    action_dim: int = 3


class EventFeaturizer:
    """Dependency-free deterministic features for replay and the tiny living core."""

    def __init__(self, dim: int = 48) -> None:
        if dim < 8:
            raise ValueError("event feature dimension must be >= 8")
        self.dim = int(dim)

    def encode(self, event: LivingEvent) -> list[float]:
        vector = [0.0] * self.dim
        text = str(event.payload.get("text", ""))
        tokens = [event.kind, event.source] + text.casefold().split()
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def time_features(dt_seconds: float) -> list[float]:
    dt = max(0.0, min(float(dt_seconds), 7 * 86400.0))
    return [
        math.log1p(dt) / math.log1p(7 * 86400.0),
        math.sin(dt / 3600.0),
        math.cos(dt / 3600.0),
        1.0 if dt >= 8 * 3600.0 else 0.0,
    ]


def analytical_parameter_count(config: LivingCoreConfig | None = None) -> int:
    c = config or LivingCoreConfig()
    input_dim = c.observed_state_dim + c.event_dim + 4
    input_layer = input_dim * c.hidden_dim + c.hidden_dim
    layer_norm = 2 * c.hidden_dim
    gru = 3 * c.latent_dim * c.hidden_dim + 3 * c.latent_dim * c.latent_dim + 6 * c.latent_dim
    state_head = c.latent_dim * c.observed_state_dim + c.observed_state_dim
    action_head = c.latent_dim * c.action_dim + c.action_dim
    return_head = c.latent_dim + 1
    return input_layer + layer_norm + gru + state_head + action_head + return_head


class TinyLivingCore:
    """Optional 14,515-parameter recurrent neural transition core.

    The model predicts bounded observed-state deltas, an action prior and a
    future-return logit while carrying a 32D latent state between events. It is
    DEVELOPMENT-only until a frozen held-out promotion court passes.
    """

    def __init__(self, config: LivingCoreConfig | None = None) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("Install neural support with: pip install -e '.[neural]'") from exc

        self.torch = torch
        self.nn = nn
        self.config = config or LivingCoreConfig()
        c = self.config

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Sequential(
                    nn.Linear(c.observed_state_dim + c.event_dim + 4, c.hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(c.hidden_dim),
                )
                self.recurrent = nn.GRUCell(c.hidden_dim, c.latent_dim)
                self.state_delta = nn.Linear(c.latent_dim, c.observed_state_dim)
                self.action = nn.Linear(c.latent_dim, c.action_dim)
                self.return_head = nn.Linear(c.latent_dim, 1)

            def forward(self, observed, event_features, dt_features, latent):
                x = self.input(torch.cat([observed, event_features, dt_features], dim=-1))
                next_latent = self.recurrent(x, latent)
                delta = 0.12 * torch.tanh(self.state_delta(next_latent))
                action_logits = self.action(next_latent)
                return_logit = self.return_head(next_latent)
                return next_latent, delta, action_logits, return_logit

        self.module = Module()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def initial_latent(self, batch_size: int = 1, *, device: str | None = None):
        return self.torch.zeros(batch_size, self.config.latent_dim, device=device)

    def __call__(self, observed, event_features, dt_features, latent):
        return self.module(observed, event_features, dt_features, latent)


def core_audit(config: LivingCoreConfig | None = None) -> dict[str, Any]:
    core = TinyLivingCore(config)
    c = core.config
    actual = core.parameter_count()
    analytical = analytical_parameter_count(c)
    if actual != analytical:
        raise RuntimeError(f"parameter audit mismatch: actual={actual} analytical={analytical}")
    return {
        "parameter_count": actual,
        "analytical_parameter_count": analytical,
        "event_dim": c.event_dim,
        "observed_state_dim": c.observed_state_dim,
        "latent_dim": c.latent_dim,
        "hidden_dim": c.hidden_dim,
        "status": "DEVELOPMENT_UNPROMOTED",
    }
