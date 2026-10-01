from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from .state_space_core import SelectiveStateSpaceCortex
from .surgery import resolve_transformer_layers


@dataclass(frozen=True, slots=True)
class CortexRegion:
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    def validate(self, *, total_layers: int) -> None:
        if self.start < 0:
            raise ValueError("cortex region start must be non-negative")
        if self.end < self.start:
            raise ValueError("cortex region end must be >= start")
        if self.width < 2:
            raise ValueError("cortex region must replace at least two blocks")
        if self.end >= int(total_layers):
            raise ValueError("cortex region exceeds decoder depth")

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


class StateSpaceRegionSession(contextlib.AbstractContextManager):
    """Collapse one contiguous decoder region into one state-space cortex call."""

    def __init__(
        self,
        model,
        cortex: SelectiveStateSpaceCortex,
        latent,
        *,
        region: CortexRegion,
        initial_state=None,
        reset_state_each_model_forward: bool = False,
    ) -> None:
        self.model = model
        self.cortex = cortex
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        region.validate(total_layers=len(self.layers))
        self.region = region
        self.initial_state = initial_state
        self.state = initial_state
        self.reset_state_each_model_forward = bool(reset_state_each_model_forward)
        self.original_forwards: dict[int, Any] = {}
        self.model_pre_handle = None
        self.model_forward_count = 0
        self.cortex_calls = 0
        self.identity_calls = {index: 0 for index in range(region.start + 1, region.end + 1)}
        self.traces: list[dict[str, Any]] = []

    def __enter__(self):
        def model_forward_hook(_module, _args, _kwargs):
            self.model_forward_count += 1
            if self.reset_state_each_model_forward:
                self.state = (
                    None
                    if self.initial_state is None
                    else self.initial_state.detach().clone()
                )

        self.model_pre_handle = self.model.register_forward_pre_hook(
            model_forward_hook,
            with_kwargs=True,
        )

        start_layer = self.layers[self.region.start]
        self.original_forwards[self.region.start] = start_layer.forward

        def cortex_forward(*args, **kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                raise ValueError("decoder layer hidden_states missing")
            updated, state, trace = self.cortex.scan(
                hidden,
                self.latent,
                state=self.state,
            )
            self.state = state
            self.cortex_calls += 1
            self.traces.append(trace)
            return updated

        start_layer.forward = cortex_forward

        for index in range(self.region.start + 1, self.region.end + 1):
            layer = self.layers[index]
            self.original_forwards[index] = layer.forward

            def identity_forward(*args, _index=index, **kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise ValueError("decoder layer hidden_states missing")
                self.identity_calls[_index] += 1
                return hidden

            layer.forward = identity_forward
        return self

    def detached_state(self):
        return None if self.state is None else self.state.detach().clone()

    def __exit__(self, exc_type, exc, tb):
        for index, original in self.original_forwards.items():
            self.layers[index].forward = original
        self.original_forwards.clear()
        if self.model_pre_handle is not None:
            self.model_pre_handle.remove()
            self.model_pre_handle = None
        return False
