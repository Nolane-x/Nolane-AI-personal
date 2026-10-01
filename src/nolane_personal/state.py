from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(slots=True)
class AffectState:
    """Slow social/affective control state, not a claim of subjective feeling."""

    valence: float = 0.0
    arousal: float = 0.35
    energy: float = 0.65
    playfulness: float = 0.45
    irritation: float = 0.0
    concern: float = 0.0
    social_drive: float = 0.20

    def normalize(self) -> None:
        self.valence = max(-1.0, min(1.0, float(self.valence)))
        for name in ("arousal", "energy", "playfulness", "irritation", "concern", "social_drive"):
            setattr(self, name, _clamp(getattr(self, name)))


@dataclass(slots=True)
class RelationshipState:
    closeness: float = 0.05
    trust: float = 0.05
    familiarity: float = 0.0
    interaction_count: int = 0

    def normalize(self) -> None:
        self.closeness = _clamp(self.closeness)
        self.trust = _clamp(self.trust)
        self.familiarity = _clamp(self.familiarity)
        self.interaction_count = max(0, int(self.interaction_count))


@dataclass(slots=True)
class OpenThread:
    thread_id: str
    topic: str
    importance: float = 0.5
    created_at: str = field(default_factory=utc_now_iso)
    last_touched_at: str = field(default_factory=utc_now_iso)
    due_at: str | None = None
    unresolved: bool = True

    def normalize(self) -> None:
        self.importance = _clamp(self.importance)
        self.topic = self.topic.strip()[:500]


@dataclass(slots=True)
class WorkingState:
    recent_topics: list[str] = field(default_factory=list)
    curiosity: float = 0.35
    uncertainty: float = 0.0
    active_intent: str | None = None

    def normalize(self) -> None:
        self.curiosity = _clamp(self.curiosity)
        self.uncertainty = _clamp(self.uncertainty)
        self.recent_topics = [str(x)[:500] for x in self.recent_topics[-8:]]


@dataclass(slots=True)
class RestState:
    cycles: int = 0
    last_cycle_at: str | None = None
    last_cycle_source_count: int = 0
    last_cycle_new_memories: int = 0

    def normalize(self) -> None:
        self.cycles = max(0, int(self.cycles))
        self.last_cycle_source_count = max(0, int(self.last_cycle_source_count))
        self.last_cycle_new_memories = max(0, int(self.last_cycle_new_memories))


@dataclass(slots=True)
class LivingState:
    schema_version: int = 2
    identity_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    tick: int = 0
    version: int = 0
    rest_mode: bool = False
    affect: AffectState = field(default_factory=AffectState)
    relationship: RelationshipState = field(default_factory=RelationshipState)
    working: WorkingState = field(default_factory=WorkingState)
    rest: RestState = field(default_factory=RestState)
    open_threads: list[OpenThread] = field(default_factory=list)
    last_event_at: str | None = None
    last_user_event_at: str | None = None
    last_ai_speech_at: str | None = None

    def normalize(self) -> None:
        self.tick = max(0, int(self.tick))
        self.version = max(0, int(self.version))
        self.affect.normalize()
        self.relationship.normalize()
        self.working.normalize()
        self.rest.normalize()
        for thread in self.open_threads:
            thread.normalize()
        self.open_threads = self.open_threads[-32:]

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LivingState":
        data = dict(payload)
        data["affect"] = AffectState(**data.get("affect", {}))
        data["relationship"] = RelationshipState(**data.get("relationship", {}))
        data["working"] = WorkingState(**data.get("working", {}))
        data["rest"] = RestState(**data.get("rest", {}))
        data["open_threads"] = [OpenThread(**item) for item in data.get("open_threads", [])]
        data["schema_version"] = max(2, int(data.get("schema_version", 1)))
        state = cls(**data)
        state.normalize()
        return state
