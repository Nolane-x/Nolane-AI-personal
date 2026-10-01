from __future__ import annotations

from .block_replacement import BlockReplacementSession
from .replacement_cortex import RecurrentReplacementCortex


class ProgressiveReplacementCortex(RecurrentReplacementCortex):
    """L10 replacement cortex with cache-aware autoregressive generation.

    Cached generation is allowed only while the first decoder layer remains
    untouched. Replacement recurrent state then persists across incremental
    one-token forwards instead of replaying old prefix state.
    """

    def generate_cached(self, **generation_inputs):
        self.assert_gradient_boundary()
        if 0 in self.config.layer_indices:
            raise ValueError("cached replacement generation cannot bypass decoder layer 0")
        self.model.eval()
        self.replacement.eval()
        generation_inputs["use_cache"] = True
        initial = self.states if self.config.carry_recurrent_state else {}
        with BlockReplacementSession(
            self.model,
            self.replacement,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=initial,
            reset_states_each_model_forward=False,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_bypass_counts = dict(session.bypass_counts)
            self.last_gate_means = {k: list(v) for k, v in session.gate_means.items()}
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
        with BlockReplacementSession(
            self.model,
            self.replacement,
            self.latent,
            layer_indices=self.config.layer_indices,
            initial_states=initial,
            reset_states_each_model_forward=True,
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_bypass_counts = dict(session.bypass_counts)
            self.last_gate_means = {k: list(v) for k, v in session.gate_means.items()}
            self.last_model_forward_count = session.model_forward_count
            if self.config.carry_recurrent_state:
                self.states = session.detached_states()
            return output
