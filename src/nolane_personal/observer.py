from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from .events import LivingEvent
from .state import LivingState


@dataclass(slots=True)
class MemoryProposal:
    text: str
    kind: str = "inference"
    confidence: float = 0.5
    salience: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThreadProposal:
    topic: str
    importance: float = 0.5
    due_at: str | None = None


@dataclass(slots=True)
class SocialProposal:
    affect_delta: dict[str, float] = field(default_factory=dict)
    relationship_delta: dict[str, float] = field(default_factory=dict)
    memories: list[MemoryProposal] = field(default_factory=list)
    open_threads: list[ThreadProposal] = field(default_factory=list)
    resolve_thread_ids: list[str] = field(default_factory=list)
    intent: str | None = None
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SocialProposal":
        return cls(
            affect_delta=dict(payload.get("affect_delta", {})),
            relationship_delta=dict(payload.get("relationship_delta", {})),
            memories=[MemoryProposal(**item) for item in payload.get("memories", [])],
            open_threads=[ThreadProposal(**item) for item in payload.get("open_threads", [])],
            resolve_thread_ids=[str(x) for x in payload.get("resolve_thread_ids", [])],
            intent=payload.get("intent"),
            uncertainty=float(payload.get("uncertainty", 0.0)),
        )


class SocialObserver(Protocol):
    def observe(self, text: str, state: LivingState, source_event: LivingEvent) -> SocialProposal: ...


class NullObserver:
    def observe(self, text: str, state: LivingState, source_event: LivingEvent) -> SocialProposal:
        return SocialProposal()


class JsonObserver:
    """Adapter for a model/function that emits one JSON object.

    The observer has no commit authority. Invalid JSON or schema raises and the
    runtime records a rejected observer event without mutating social state.
    """

    def __init__(self, generate_json: Callable[[str], str]) -> None:
        self.generate_json = generate_json

    def observe(self, text: str, state: LivingState, source_event: LivingEvent) -> SocialProposal:
        prompt = {
            "task": "propose_social_state_update",
            "source_event_id": source_event.event_id,
            "user_text": text,
            "current_state": {
                "affect": asdict(state.affect),
                "relationship": asdict(state.relationship),
                "working": asdict(state.working),
                "open_threads": [asdict(t) for t in state.open_threads if t.unresolved],
            },
            "rules": {
                "output": "JSON object only",
                "uncertainty": "0..1",
                "never_assert_unobserved_fact": True,
                "proposal_only": True,
            },
        }
        raw = self.generate_json(json.dumps(prompt, ensure_ascii=False))
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("observer output must be a JSON object")
        return SocialProposal.from_dict(payload)
