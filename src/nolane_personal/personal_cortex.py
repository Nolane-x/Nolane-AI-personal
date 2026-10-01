from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .surgery import LatentHookSession, LatentResidualAdapter, parameter_guard_snapshot


@dataclass(slots=True)
class PersonalCortexConfig:
    layer_indices: tuple[int, ...] | None = None
    token_scope: str = "all"

    def validate(self) -> None:
        if self.token_scope not in {"last", "all"}:
            raise ValueError("token_scope must be 'last' or 'all'")
        if self.layer_indices is not None and not self.layer_indices:
            raise ValueError("layer_indices cannot be empty")


class TrainablePersonalCortex:
    """Qwen + persistent-latent adapter with strict gradient ownership.

    Base-model parameters are frozen. The only trainable neural parameters are
    owned by the latent adapter. Unlike L5 shadow probing, this class is a real
    model path: its forward/generate calls run through latent-conditioned
    decoder hidden states.
    """

    def __init__(
        self,
        model,
        adapter: LatentResidualAdapter,
        latent,
        *,
        config: PersonalCortexConfig | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.latent = latent
        self.config = config or PersonalCortexConfig()
        self.config.validate()
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def trainable_parameters(self):
        return [p for p in self.adapter.module.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base model gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("adapter exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.model)

    def forward(self, **model_inputs):
        self.assert_gradient_boundary()
        with LatentHookSession(
            self.model,
            self.adapter,
            self.latent,
            layer_indices=self.config.layer_indices,
            gate_override=None,
            token_scope=self.config.token_scope,
        ):
            return self.model(**model_inputs)

    def generate(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.adapter.eval()
        with LatentHookSession(
            self.model,
            self.adapter,
            self.latent,
            layer_indices=self.config.layer_indices,
            gate_override=None,
            token_scope="last",
        ):
            return self.model.generate(**generation_inputs)

    def adapter_state_dict(self) -> dict[str, Any]:
        return self.adapter.module.state_dict()
