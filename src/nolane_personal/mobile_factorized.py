from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA = "NOLANE-V050-MOBILE-FACTORIZED-TOKEN-STEP-V1"


@dataclass(frozen=True, slots=True)
class MobileFactorizedContract:
    vocab_size: int
    hidden_size: int
    rank: int
    latent_dim: int
    state_dim: int
    packed_state_dim: int
    virtual_steps: int
    tie_word_embeddings: bool
    rms_norm_eps: float
    slow_decay_floor: float
    max_abs_gate: float
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            **asdict(self),
            "input_contract": {
                "init_latent": [1, self.latent_dim],
                "step_token_id": [1],
                "step_state": [1, self.packed_state_dim],
                "step_latent": [1, self.latent_dim],
            },
            "output_contract": {
                "init_state": [1, self.packed_state_dim],
                "step_logits": [1, self.vocab_size],
                "step_state": [1, self.packed_state_dim],
            },
            "autoregressive_loop_in_graph": False,
            "sampling_in_graph": False,
            "dynamic_eos_loop_in_graph": False,
        }


def _frozen_parameter(nn, tensor):
    return nn.Parameter(tensor.detach().clone(), requires_grad=False)


class MobileFactorizedInitState:
    """Factory namespace for the exportable init-state nn.Module."""

    @staticmethod
    def from_model(model):
        torch = model.cortex.torch
        nn = model.cortex.nn
        module = model.cortex.module
        config = model.cortex.config

        class InitState(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.latent_norm_weight = _frozen_parameter(
                    nn,
                    module.latent_norm.weight,
                )
                self.latent_proj_weight = _frozen_parameter(
                    nn,
                    module.latent_proj.weight,
                )
                self.latent_proj_bias = _frozen_parameter(
                    nn,
                    module.latent_proj.bias,
                )
                self.latent_eps = float(module.latent_norm.eps)

            def forward(self, latent):
                normalized = latent * torch.rsqrt(
                    latent.pow(2).mean(dim=-1, keepdim=True)
                    + self.latent_eps
                ) * self.latent_norm_weight
                feature = torch.tanh(
                    torch.nn.functional.linear(
                        normalized,
                        self.latent_proj_weight,
                        self.latent_proj_bias,
                    )
                )
                return torch.cat(
                    [feature, feature, feature],
                    dim=-1,
                )

        return InitState().eval()


class MobileFactorizedTokenStep:
    """Factory namespace for the exportable one-token factorized graph.

    The graph intentionally contains no autoregressive loop and no sampling.
    Android/native code owns token iteration, EOS handling and sampling.
    """

    @staticmethod
    def from_model(model):
        torch = model.cortex.torch
        nn = model.cortex.nn
        boundary = model.boundary
        cortex = model.cortex
        c = cortex.config
        b = boundary.config
        cm = cortex.module

        class TokenStep(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.input_codes = _frozen_parameter(
                    nn,
                    boundary.embed_tokens.codes.weight,
                )
                self.input_basis = _frozen_parameter(
                    nn,
                    boundary.embed_tokens.basis,
                )
                self.final_norm_weight = _frozen_parameter(
                    nn,
                    boundary.final_norm.weight,
                )

                if b.tie_word_embeddings:
                    self.output_codes = None
                    self.output_basis = None
                else:
                    self.output_codes = _frozen_parameter(
                        nn,
                        boundary.output_codes,
                    )
                    self.output_basis = _frozen_parameter(
                        nn,
                        boundary.output_basis,
                    )

                self.hidden_norm_weight = _frozen_parameter(
                    nn,
                    cm.hidden_norm.weight,
                )
                self.hidden_norm_bias = _frozen_parameter(
                    nn,
                    cm.hidden_norm.bias,
                )
                self.hidden_down_weight = _frozen_parameter(
                    nn,
                    cm.hidden_down.weight,
                )
                self.hidden_down_bias = _frozen_parameter(
                    nn,
                    cm.hidden_down.bias,
                )

                self.latent_norm_weight = _frozen_parameter(
                    nn,
                    cm.latent_norm.weight,
                )
                self.latent_proj_weight = _frozen_parameter(
                    nn,
                    cm.latent_proj.weight,
                )
                self.latent_proj_bias = _frozen_parameter(
                    nn,
                    cm.latent_proj.bias,
                )

                for name in (
                    "fast_proposal",
                    "fast_decay",
                    "slow_proposal",
                    "slow_decay",
                    "fast_output_gate",
                    "slow_output_gate",
                    "depth_transition",
                    "depth_token",
                    "depth_gate",
                    "state_out",
                ):
                    layer = getattr(cm, name)
                    setattr(
                        self,
                        name + "_weight",
                        _frozen_parameter(nn, layer.weight),
                    )
                    setattr(
                        self,
                        name + "_bias",
                        _frozen_parameter(nn, layer.bias),
                    )

                self.depth_embedding = _frozen_parameter(
                    nn,
                    cm.depth_embedding,
                )
                self.raw_gate = _frozen_parameter(
                    nn,
                    cm.raw_gate,
                )

                self.hidden_size = int(b.hidden_size)
                self.state_dim = int(c.state_dim)
                self.virtual_steps = int(c.virtual_steps)
                self.hidden_norm_eps = float(cm.hidden_norm.eps)
                self.latent_norm_eps = float(cm.latent_norm.eps)
                self.final_norm_eps = float(b.rms_norm_eps)
                self.slow_decay_floor = float(c.slow_decay_floor)
                self.max_abs_gate = float(c.max_abs_gate)
                self.tie_word_embeddings = bool(
                    b.tie_word_embeddings
                )

            @staticmethod
            def _linear(x, weight, bias):
                return torch.nn.functional.linear(
                    x,
                    weight,
                    bias,
                )

            def _latent_feature(self, latent):
                normalized = latent * torch.rsqrt(
                    latent.pow(2).mean(dim=-1, keepdim=True)
                    + self.latent_norm_eps
                ) * self.latent_norm_weight
                return torch.tanh(
                    self._linear(
                        normalized,
                        self.latent_proj_weight,
                        self.latent_proj_bias,
                    )
                )

            def _final_norm(self, hidden):
                source_dtype = hidden.dtype
                states = hidden.float()
                states = states * torch.rsqrt(
                    states.pow(2).mean(dim=-1, keepdim=True)
                    + self.final_norm_eps
                )
                return self.final_norm_weight * states.to(source_dtype)

            def forward(self, token_id, state, latent):
                rank_code = torch.nn.functional.embedding(
                    token_id,
                    self.input_codes,
                )
                hidden = rank_code @ self.input_basis

                normalized_hidden = torch.nn.functional.layer_norm(
                    hidden,
                    (self.hidden_size,),
                    self.hidden_norm_weight,
                    self.hidden_norm_bias,
                    self.hidden_norm_eps,
                )
                token = torch.tanh(
                    self._linear(
                        normalized_hidden.to(
                            dtype=self.hidden_down_weight.dtype
                        ),
                        self.hidden_down_weight,
                        self.hidden_down_bias,
                    )
                )
                latent_feature = self._latent_feature(latent)

                d = self.state_dim
                fast = state[:, :d]
                slow = state[:, d : 2 * d]
                depth = state[:, 2 * d : 3 * d]

                fast_proposal = torch.tanh(
                    self._linear(
                        token,
                        self.fast_proposal_weight,
                        self.fast_proposal_bias,
                    )
                    + latent_feature
                )
                fast_decay = torch.sigmoid(
                    self._linear(
                        token,
                        self.fast_decay_weight,
                        self.fast_decay_bias,
                    )
                )
                fast = (
                    fast_decay * fast
                    + (1.0 - fast_decay) * fast_proposal
                )

                slow_proposal = torch.tanh(
                    self._linear(
                        fast,
                        self.slow_proposal_weight,
                        self.slow_proposal_bias,
                    )
                    + latent_feature
                )
                raw_slow_decay = torch.sigmoid(
                    self._linear(
                        token,
                        self.slow_decay_weight,
                        self.slow_decay_bias,
                    )
                )
                slow_decay = (
                    self.slow_decay_floor
                    + (1.0 - self.slow_decay_floor)
                    * raw_slow_decay
                )
                slow = (
                    slow_decay * slow
                    + (1.0 - slow_decay) * slow_proposal
                )

                depth = (
                    0.5 * depth
                    + 0.25 * fast
                    + 0.25 * slow
                )
                for step_index in range(self.virtual_steps):
                    step_feature = self.depth_embedding[
                        step_index
                    ].unsqueeze(0)
                    proposal = torch.tanh(
                        self._linear(
                            depth,
                            self.depth_transition_weight,
                            self.depth_transition_bias,
                        )
                        + self._linear(
                            token,
                            self.depth_token_weight,
                            self.depth_token_bias,
                        )
                        + latent_feature
                        + step_feature
                    )
                    carry = torch.sigmoid(
                        self._linear(
                            depth + step_feature,
                            self.depth_gate_weight,
                            self.depth_gate_bias,
                        )
                    )
                    depth = (
                        carry * depth
                        + (1.0 - carry) * proposal
                    )

                fast_gate = torch.sigmoid(
                    self._linear(
                        token,
                        self.fast_output_gate_weight,
                        self.fast_output_gate_bias,
                    )
                )
                slow_gate = torch.sigmoid(
                    self._linear(
                        token,
                        self.slow_output_gate_weight,
                        self.slow_output_gate_bias,
                    )
                )
                joined = torch.cat(
                    [
                        fast * fast_gate,
                        slow * slow_gate,
                        depth,
                    ],
                    dim=-1,
                )
                residual = self._linear(
                    joined,
                    self.state_out_weight,
                    self.state_out_bias,
                )
                gate = self.max_abs_gate * torch.tanh(
                    self.raw_gate
                )
                updated = hidden + gate.to(
                    dtype=hidden.dtype
                ) * residual.to(dtype=hidden.dtype)

                normalized = self._final_norm(updated)
                if self.tie_word_embeddings:
                    rank_hidden = normalized @ self.input_basis.t()
                    logits = rank_hidden @ self.input_codes.t()
                else:
                    rank_hidden = normalized @ self.output_basis.t()
                    logits = rank_hidden @ self.output_codes.t()

                next_state = torch.cat(
                    [fast, slow, depth],
                    dim=-1,
                )
                return logits.float(), next_state

        return TokenStep().eval()


def mobile_factorized_contract(model) -> MobileFactorizedContract:
    boundary = model.boundary.config
    cortex = model.cortex.config
    return MobileFactorizedContract(
        vocab_size=int(boundary.vocab_size),
        hidden_size=int(boundary.hidden_size),
        rank=int(boundary.rank),
        latent_dim=int(cortex.latent_dim),
        state_dim=int(cortex.state_dim),
        packed_state_dim=int(3 * cortex.state_dim),
        virtual_steps=int(cortex.virtual_steps),
        tie_word_embeddings=bool(boundary.tie_word_embeddings),
        rms_norm_eps=float(boundary.rms_norm_eps),
        slow_decay_floor=float(cortex.slow_decay_floor),
        max_abs_gate=float(cortex.max_abs_gate),
        bos_token_id=boundary.bos_token_id,
        eos_token_id=boundary.eos_token_id,
        pad_token_id=boundary.pad_token_id,
    )


def build_mobile_factorized_modules(model):
    return (
        MobileFactorizedInitState.from_model(model),
        MobileFactorizedTokenStep.from_model(model),
        mobile_factorized_contract(model),
    )
