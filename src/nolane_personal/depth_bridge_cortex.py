from __future__ import annotations

from dataclasses import dataclass

from .depth_bridge import CrossLayerLivingBridge, LivingBridgeHookSession
from .surgery import parameter_guard_snapshot


@dataclass(slots=True)
class LivingBridgeCortexConfig:
    layer_indices: tuple[int, ...] | None = None
    token_scope: str = "all"

    def validate(self) -> None:
        if self.token_scope not in {"all", "last"}:
            raise ValueError("token_scope must be 'all' or 'last'")
        if self.layer_indices is not None and not self.layer_indices:
            raise ValueError("layer_indices cannot be empty")


class TrainableLivingBridgeCortex:
    def __init__(self, model, bridge: CrossLayerLivingBridge, latent, *, config=None) -> None:
        self.model = model
        self.bridge = bridge
        self.latent = latent
        self.config = config or LivingBridgeCortexConfig()
        self.config.validate()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_latent(self, latent) -> None:
        self.latent = latent

    def trainable_parameters(self):
        return [p for p in self.bridge.module.parameters() if p.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def assert_gradient_boundary(self) -> None:
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("base Qwen gradient boundary violated")
        if not self.trainable_parameters():
            raise RuntimeError("Living Bridge exposes no trainable parameters")

    def base_guard(self):
        return parameter_guard_snapshot(self.model)

    def forward(self, **model_inputs):
        self.assert_gradient_boundary()
        with LivingBridgeHookSession(
            self.model,
            self.bridge,
            self.latent,
            layer_indices=self.config.layer_indices,
            token_scope=self.config.token_scope,
        ) as session:
            output = self.model(**model_inputs)
            self.last_trace = list(session.trace)
            return output

    def generate(self, **generation_inputs):
        self.assert_gradient_boundary()
        self.model.eval()
        self.bridge.eval()
        with LivingBridgeHookSession(
            self.model,
            self.bridge,
            self.latent,
            layer_indices=self.config.layer_indices,
            token_scope="last",
        ) as session:
            output = self.model.generate(**generation_inputs)
            self.last_trace = list(session.trace)
            return output
