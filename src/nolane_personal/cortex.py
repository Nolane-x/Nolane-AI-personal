from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .memory import MemoryRecord
from .state import LivingState


@dataclass(slots=True)
class CortexRequest:
    mode: str
    intent: str
    user_text: str | None
    state: LivingState
    memories: list[MemoryRecord] = field(default_factory=list)
    recent_messages: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class CortexReply:
    utterance: str
    intent: str = "conversation"


class Cortex(Protocol):
    def generate(self, request: CortexRequest) -> CortexReply: ...


class NullCortex:
    def generate(self, request: CortexRequest) -> CortexReply:
        return CortexReply("", intent=request.intent)
