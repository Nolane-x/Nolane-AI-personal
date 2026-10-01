from __future__ import annotations

from dataclasses import dataclass

from .state_space_core import SelectiveStateSpaceCortex
from .state_space_region import CortexRegion, StateSpaceRegionSession
from .surgery import parameter_guard_snapshot, resolve_transformer_layers


@dataclass(slots=True)
class StateSpaceModelConfig:
    region: CortexRegion
    carry_recurrent_state: bool = False

    def validate(self, *, total_layers: int) -> None:
        self.region.validate(total_layers=total_layers)


class StateSpacePersonalModel:
    """Frozen Qwen with one wide decoder region replaced by an SSM cortex."""

    def __init__(
        self,
        model,
        cortex: SelectiveStateSpaceCortex,
        latent,
        *,
        config: StateSpaceModelConfig,
    ) -> None:
        self.model = model
        self.cortex = cortex
        self.latent = latent
        self.config = config
        self.config.validate(total_layers=len(resolve_transformer_layers(model)))
        self.state = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def reset_state(self) -> None:
        self.state = None

    def trainable_parameters(self):
        return [p for p in self.cortex.module.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def replaced_layer_count(self) -> int:
        return self.config.region.width

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("state-space cortex exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.model)

    def forward(self, **model_inputs):
        self.assert_gradient_boundary()
        initial = self.state if self.config.carry_recurrent_state else None
        with StateSpaceRegionSession(
            self.model,
            self.cortex,
            self.latent,
            region=self.config.region,
            initial_state=initial,
        ) as session:
            output = self.model(**model_inputs)
            self.last_cortex_calls = session.cortex_calls
            self.last_identity_calls = dict(session.identity_calls)
            self.last_traces = list(session.traces)
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.state = session.detached_state()
            return output

    def generate_cached(self, **generation_inputs):
        self.assert_gradient_boundary()
        if self.config.region.start == 0:
            raise ValueError("cached state-space generation cannot replace decoder layer 0")
        self.model.eval()
        self.cortex.eval()
        generation_inputs["use_cache"] = True
        initial = self.state if self.config.carry_recurrent_state else None
        with StateSpaceRegionSession(
            self.model,
            self.cortex,
            self.latent,
            region=self.config.region,
            initial_state=initial,
            reset_state_each_model_forward=False,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_cortex_calls = session.cortex_calls
            self.last_identity_calls = dict(session.identity_calls)
            self.last_traces = list(session.traces)
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.state = session.detached_state()
            return output

    def generate_replay_safe(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.cortex.eval()
        generation_inputs["use_cache"] = False
        initial = self.state if self.config.carry_recurrent_state else None
        with StateSpaceRegionSession(
            self.model,
            self.cortex,
            self.latent,
            region=self.config.region,
            initial_state=initial,
            reset_state_each_model_forward=True,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_cortex_calls = session.cortex_calls
            self.last_identity_calls = dict(session.identity_calls)
            self.last_traces = list(session.traces)
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.state = session.detached_state()
            return output
