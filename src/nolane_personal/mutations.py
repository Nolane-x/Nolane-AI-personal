from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from .events import LivingEvent
from .memory import MemoryRecord
from .observer import SocialProposal
from .state import LivingState, OpenThread


AFFECT_FIELDS = {"valence", "arousal", "energy", "playfulness", "irritation", "concern", "social_drive"}
RELATIONSHIP_FIELDS = {"closeness", "trust", "familiarity"}


@dataclass(slots=True)
class MutationPolicy:
    max_affect_delta: float = 0.12
    max_relationship_delta: float = 0.035
    max_memories_per_event: int = 4
    max_open_threads_per_event: int = 3
    max_memory_chars: int = 800
    fact_confidence_floor: float = 0.90
    max_intent_chars: int = 120


@dataclass(slots=True)
class MutationReceipt:
    source_event_id: str
    accepted_affect_delta: dict[str, float] = field(default_factory=dict)
    accepted_relationship_delta: dict[str, float] = field(default_factory=dict)
    stored_memory_ids: list[str] = field(default_factory=list)
    opened_thread_ids: list[str] = field(default_factory=list)
    resolved_thread_ids: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    downgraded_memories: int = 0
    uncertainty: float = 0.0
    intent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MutationValidator:
    """Single commit boundary for model-proposed persistent changes."""

    def __init__(self, policy: MutationPolicy | None = None) -> None:
        self.policy = policy or MutationPolicy()

    @staticmethod
    def _bounded(value: float, limit: float) -> float:
        return max(-limit, min(limit, float(value)))

    @staticmethod
    def _unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def apply(
        self,
        state: LivingState,
        proposal: SocialProposal,
        source_event: LivingEvent,
    ) -> tuple[LivingState, list[MemoryRecord], MutationReceipt]:
        next_state = deepcopy(state)
        receipt = MutationReceipt(source_event_id=source_event.event_id)
        receipt.uncertainty = self._unit(proposal.uncertainty)

        for name, raw in proposal.affect_delta.items():
            if name not in AFFECT_FIELDS:
                receipt.rejected.append(f"affect:{name}")
                continue
            delta = self._bounded(raw, self.policy.max_affect_delta)
            setattr(next_state.affect, name, getattr(next_state.affect, name) + delta)
            receipt.accepted_affect_delta[name] = delta

        for name, raw in proposal.relationship_delta.items():
            if name not in RELATIONSHIP_FIELDS:
                receipt.rejected.append(f"relationship:{name}")
                continue
            delta = self._bounded(raw, self.policy.max_relationship_delta)
            setattr(next_state.relationship, name, getattr(next_state.relationship, name) + delta)
            receipt.accepted_relationship_delta[name] = delta

        memories: list[MemoryRecord] = []
        for item in proposal.memories[: self.policy.max_memories_per_event]:
            text = item.text.strip()[: self.policy.max_memory_chars]
            if not text:
                receipt.rejected.append("memory:empty")
                continue
            confidence = self._unit(item.confidence)
            kind = item.kind if item.kind in {"episodic", "fact", "preference", "inference"} else "inference"
            metadata = dict(item.metadata)
            if kind == "fact" and confidence < self.policy.fact_confidence_floor:
                kind = "inference"
                metadata["downgraded_from_fact"] = True
                receipt.downgraded_memories += 1
            metadata["observer_uncertainty"] = receipt.uncertainty
            metadata["proposed_by_observer"] = True
            memory = MemoryRecord(
                text=text,
                kind=kind,
                confidence=confidence,
                salience=self._unit(item.salience),
                source_event_id=source_event.event_id,
                created_at=source_event.at,
                metadata=metadata,
            )
            memories.append(memory)
            receipt.stored_memory_ids.append(memory.memory_id)

        for item in proposal.open_threads[: self.policy.max_open_threads_per_event]:
            topic = item.topic.strip()[:500]
            if not topic:
                receipt.rejected.append("thread:empty")
                continue
            thread = OpenThread(
                thread_id=f"obs-{source_event.event_id[:8]}-{len(next_state.open_threads)+1}",
                topic=topic,
                importance=self._unit(item.importance),
                due_at=item.due_at,
                created_at=source_event.at,
                last_touched_at=source_event.at,
            )
            next_state.open_threads.append(thread)
            receipt.opened_thread_ids.append(thread.thread_id)

        wanted = set(proposal.resolve_thread_ids[:8])
        for thread in next_state.open_threads:
            if thread.thread_id in wanted and thread.unresolved:
                thread.unresolved = False
                thread.last_touched_at = source_event.at
                receipt.resolved_thread_ids.append(thread.thread_id)

        unresolved_requested = wanted - set(receipt.resolved_thread_ids)
        for thread_id in sorted(unresolved_requested):
            receipt.rejected.append(f"resolve_missing:{thread_id}")

        if proposal.intent:
            intent = str(proposal.intent).strip()[: self.policy.max_intent_chars]
            next_state.working.active_intent = intent or None
            receipt.intent = intent or None
        next_state.working.uncertainty = receipt.uncertainty

        next_state.updated_at = source_event.at
        next_state.last_event_at = source_event.at
        next_state.normalize()
        return next_state, memories, receipt
