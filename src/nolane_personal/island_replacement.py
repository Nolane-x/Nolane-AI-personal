from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .block_replacement import RecurrentBlockReplacement
from .surgery import resolve_transformer_layers


@dataclass(frozen=True, slots=True, order=True)
class TransformerIsland:
    start: int
    end: int

    def validate(self, *, total_layers: int | None = None) -> None:
        if self.start < 0:
            raise ValueError("island start must be non-negative")
        if self.end < self.start:
            raise ValueError("island end must be >= start")
        if self.width < 2:
            raise ValueError("island must replace at least two contiguous blocks")
        if total_layers is not None and self.end >= int(total_layers):
            raise ValueError("island exceeds decoder depth")

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_islands(
    islands: Iterable[TransformerIsland | tuple[int, int] | list[int]],
    *,
    total_layers: int,
    min_gap_layers: int = 0,
) -> tuple[TransformerIsland, ...]:
    normalized = []
    for raw in islands:
        island = raw if isinstance(raw, TransformerIsland) else TransformerIsland(int(raw[0]), int(raw[1]))
        island.validate(total_layers=total_layers)
        normalized.append(island)
    normalized.sort()
    if not normalized:
        raise ValueError("at least one Transformer island is required")
    for left, right in zip(normalized, normalized[1:]):
        gap = right.start - left.end - 1
        if gap < int(min_gap_layers):
            raise ValueError("Transformer islands overlap or violate minimum gap")
    return tuple(normalized)


class TransformerIslandSession(contextlib.AbstractContextManager):
    """Replace whole contiguous Qwen regions with one recurrent call per island.

    The first decoder block of each island invokes the recurrent replacement.
    Every later decoder block inside that island becomes an identity mapping.
    Therefore an island of width N executes the replacement once instead of
    executing N original attention+MLP blocks.
    """

    def __init__(
        self,
        model,
        replacement: RecurrentBlockReplacement,
        latent,
        *,
        islands: Iterable[TransformerIsland | tuple[int, int] | list[int]],
        initial_states: dict[int, Any] | None = None,
        reset_states_each_model_forward: bool = False,
    ) -> None:
        self.model = model
        self.replacement = replacement
        self.latent = latent
        self.layers = resolve_transformer_layers(model)
        self.islands = normalize_islands(islands, total_layers=len(self.layers))
        if any(island.start >= replacement.config.max_layers for island in self.islands):
            raise ValueError("island identity exceeds replacement max_layers")
        self.initial_states = dict(initial_states or {})
        self.states = dict(self.initial_states)
        self.reset_states_each_model_forward = bool(reset_states_each_model_forward)
        self.model_pre_handle = None
        self.model_forward_count = 0
        self.original_forwards: dict[int, Any] = {}
        self.island_calls: dict[int, int] = {island.start: 0 for island in self.islands}
        self.identity_calls: dict[int, int] = {
            index: 0
            for island in self.islands
            for index in range(island.start + 1, island.end + 1)
        }
        self.gate_means: dict[int, list[float]] = {island.start: [] for island in self.islands}

    def __enter__(self):
        def model_forward_hook(_module, _args, _kwargs):
            self.model_forward_count += 1
            if self.reset_states_each_model_forward:
                self.states = {
                    start: state.detach().clone()
                    for start, state in self.initial_states.items()
                }

        self.model_pre_handle = self.model.register_forward_pre_hook(
            model_forward_hook,
            with_kwargs=True,
        )

        for island in self.islands:
            start_layer = self.layers[island.start]
            self.original_forwards[island.start] = start_layer.forward

            def island_forward(*args, _start=island.start, **kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise ValueError("decoder layer hidden_states missing")
                replaced, state, gate = self.replacement.replace(
                    hidden,
                    self.latent,
                    layer_index=_start,
                    state=self.states.get(_start),
                )
                self.states[_start] = state
                self.island_calls[_start] += 1
                self.gate_means[_start].append(float(gate.detach().float().mean().cpu()))
                return replaced

            start_layer.forward = island_forward

            for index in range(island.start + 1, island.end + 1):
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

    @property
    def replaced_layer_count(self) -> int:
        return sum(island.width for island in self.islands)

    @property
    def island_count(self) -> int:
        return len(self.islands)

    def detached_states(self) -> dict[int, Any]:
        return {start: state.detach().clone() for start, state in self.states.items()}

    def __exit__(self, exc_type, exc, tb):
        for index, original in self.original_forwards.items():
            self.layers[index].forward = original
        self.original_forwards.clear()
        if self.model_pre_handle is not None:
            self.model_pre_handle.remove()
            self.model_pre_handle = None
        return False
