from __future__ import annotations

from dataclasses import dataclass

from .block_replacement import BlockReplacementSession, RecurrentBlockReplacement
from .surgery import parameter_guard_snapshot


@dataclass(slots=True)
class ReplacementCortexConfig:
    layer_indices: tuple[int, ...]
    carry_recurrent_state: bool = False

    def validate(self) -> None:
        if not self.layer_indices:
            raise ValueError("at least one decoder layer must be replaced")


class RecurrentReplacementCortex:
    """Frozen Qwen with selected decoder blocks bypassed by a recurrent substitute."""

    def __init__(self, model, replacement: RecurrentBlockReplacement, latent, *, config: ReplacementCortexConfig) -> None:
        self.model = model
        self.replacement = replacement
        self.latent = latent
        self.config = config
        self.config.validate()
        self.states = {}
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_state(self) -> None:
        self.states = {}

    def trainable_parameters(self):
        return [p for p in self.replacement.module.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("replacement exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.model)

    def forward(self, **model_inputs):
        self.assert_gradient_boundary()
        initial = self.states if self.config.carry_recurrent_state else {}
        with BlockReplacementSession(
            self.model,
            self.replacement,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=initial,
        ) as session:
            output = self.model(**model_inputs)
            self.last_bypass_counts = dict(session.bypass_counts)
            self.last_gate_means = {k: list(v) for k, v in session.gate_means.items()}
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output

    def generate(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.replacement.eval()
        generation_inputs.setdefault("use_cache", False)
        initial = self.states if self.config.carry_recurrent_state else {}
        with BlockReplacementSession(
            self.model,
            self.replacement,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=initial,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_bypass_counts = dict(session.bypass_counts)
            self.last_gate_means = {k: list(v) for k, v in session.gate_means.items()}
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output
