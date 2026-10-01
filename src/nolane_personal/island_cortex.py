from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .block_replacement import RecurrentBlockReplacement
from .island_replacement import TransformerIsland, TransformerIslandSession, normalize_islands
from .surgery import parameter_guard_snapshot, resolve_transformer_layers


@dataclass(slots=True)
class IslandCortexConfig:
    islands: tuple[TransformerIsland, ...]
    carry_recurrent_state: bool = False

    def validate(self, *, total_layers: int | None = None) -> None:
        if not self.islands:
            raise ValueError("at least one Transformer island is required")
        if total_layers is not None:
            normalize_islands(self.islands, total_layers=total_layers)


class RecurrentIslandCortex:
    """Frozen Qwen where contiguous decoder regions collapse to recurrent islands."""

    def __init__(
        self,
        model,
        replacement: RecurrentBlockReplacement,
        latent,
        *,
        config: IslandCortexConfig,
    ) -> None:
        self.model = model
        self.replacement = replacement
        self.latent = latent
        self.config = config
        self.config.validate(total_layers=len(resolve_transformer_layers(model)))
        self.states: dict[int, object] = {}
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_state(self) -> None:
        self.states = {}

    def trainable_parameters(self):
        return [parameter for parameter in self.replacement.module.parameters() if parameter.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def frozen_base_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def replaced_layer_count(self) -> int:
        return sum(island.width for island in self.config.islands)

    def assert_gradient_boundary(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("base Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("island replacement exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.model)

    def forward(self, **model_inputs):
        self.assert_gradient_boundary()
        initial = self.states if self.config.carry_recurrent_state else {}
        with TransformerIslandSession(
            self.model,
            self.replacement,
            self.latent,
            islands=self.config.islands,
            initial_states=initial,
        ) as session:
            output = self.model(**model_inputs)
            self.last_island_calls = dict(session.island_calls)
            self.last_identity_calls = dict(session.identity_calls)
            self.last_gate_means = {key: list(values) for key, values in session.gate_means.items()}
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output

    def generate_cached(self, **generation_inputs):
        self.assert_gradient_boundary()
        if any(island.start == 0 for island in self.config.islands):
            raise ValueError("cached island generation cannot replace decoder layer 0")
        self.model.eval()
        self.replacement.eval()
        generation_inputs["use_cache"] = True
        initial = self.states if self.config.carry_recurrent_state else {}
        with TransformerIslandSession(
            self.model,
            self.replacement,
            self.latent,
            islands=self.config.islands,
            initial_states=initial,
            reset_states_each_model_forward=False,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_island_calls = dict(session.island_calls)
            self.last_identity_calls = dict(session.identity_calls)
            self.last_gate_means = {key: list(values) for key, values in session.gate_means.items()}
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output

    def generate_replay_safe(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.replacement.eval()
        generation_inputs["use_cache"] = False
        initial = self.states if self.config.carry_recurrent_state else {}
        with TransformerIslandSession(
            self.model,
            self.replacement,
            self.latent,
            islands=self.config.islands,
            initial_states=initial,
            reset_states_each_model_forward=True,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_island_calls = dict(session.island_calls)
            self.last_identity_calls = dict(session.identity_calls)
            self.last_gate_means = {key: list(values) for key, values in session.gate_means.items()}
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output
