from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from .events import LivingEvent
from .memory import MemoryRecord
from .rest import RestProposal
from .state import LivingState


@dataclass(slots=True)
class MemoryLink:
    parent_memory_id: str
    child_memory_id: str
    relation: str
    source_event_id: str
    created_at: str


@dataclass(slots=True)
class ConsolidationPolicy:
    max_new_memories: int = 4
    max_sources_per_memory: int = 8
    min_sources: int = 2
    max_text_chars: int = 1000
    fact_confidence_floor: float = 0.95
    max_thread_resolutions: int = 2
    max_intent_chars: int = 120


@dataclass(slots=True)
class ConsolidationReceipt:
    source_event_id: str
    stored_memory_ids: list[str] = field(default_factory=list)
    memory_links: int = 0
    resolved_thread_ids: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    downgraded_memories: int = 0
    active_intent: str | None = None
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConsolidationValidator:
    """Commit boundary for REST proposals.

    REST may derive new memories from existing evidence, but cannot fabricate a
    source, erase evidence, change relationship state, or mutate personality.
    """

    ALLOWED_KINDS = {"episodic", "preference", "inference", "fact", "habit"}

    def __init__(self, policy: ConsolidationPolicy | None = None) -> None:
        self.policy = policy or ConsolidationPolicy()

    @staticmethod
    def _unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def apply(
        self,
        state: LivingState,
        proposal: RestProposal,
        source_event: LivingEvent,
        source_memories: dict[str, MemoryRecord],
    ) -> tuple[LivingState, list[MemoryRecord], list[MemoryLink], ConsolidationReceipt]:
        next_state = deepcopy(state)
        receipt = ConsolidationReceipt(source_event_id=source_event.event_id)
        receipt.uncertainty = self._unit(proposal.uncertainty)
        new_memories: list[MemoryRecord] = []
        links: list[MemoryLink] = []

        for item in proposal.consolidated_memories[: self.policy.max_new_memories]:
            source_ids = list(dict.fromkeys(item.source_memory_ids[: self.policy.max_sources_per_memory]))
            if len(source_ids) < self.policy.min_sources:
                receipt.rejected.append("memory:insufficient_sources")
                continue
            if any(memory_id not in source_memories for memory_id in source_ids):
                receipt.rejected.append("memory:unknown_source")
                continue
            text = item.text.strip()[: self.policy.max_text_chars]
            if not text:
                receipt.rejected.append("memory:empty")
                continue

            sources = [source_memories[memory_id] for memory_id in source_ids]
            confidence = min(self._unit(item.confidence), min(m.confidence for m in sources))
            kind = item.kind if item.kind in self.ALLOWED_KINDS else "inference"
            metadata = dict(item.metadata)
            if kind == "fact":
                all_fact = all(m.kind == "fact" and m.confidence >= self.policy.fact_confidence_floor for m in sources)
                if not all_fact or confidence < self.policy.fact_confidence_floor:
                    kind = "inference"
                    metadata["downgraded_from_fact"] = True
                    receipt.downgraded_memories += 1

            metadata.update({
                "rest_consolidated": True,
                "source_memory_ids": source_ids,
                "rest_uncertainty": receipt.uncertainty,
            })
            memory = MemoryRecord(
                text=text,
                kind=kind,
                confidence=confidence,
                salience=self._unit(item.salience),
                source_event_id=source_event.event_id,
                created_at=source_event.at,
                metadata=metadata,
            )
            new_memories.append(memory)
            receipt.stored_memory_ids.append(memory.memory_id)
            for parent_id in source_ids:
                links.append(
                    MemoryLink(
                        parent_memory_id=parent_id,
                        child_memory_id=memory.memory_id,
                        relation="consolidated_into",
                        source_event_id=source_event.event_id,
                        created_at=source_event.at,
                    )
                )

        allowed_resolutions = 0
        for review in proposal.thread_reviews:
            if review.action != "resolve":
                continue
            if allowed_resolutions >= self.policy.max_thread_resolutions:
                receipt.rejected.append(f"thread_resolution_limit:{review.thread_id}")
                continue
            for thread in next_state.open_threads:
                if thread.thread_id == review.thread_id and thread.unresolved:
                    thread.unresolved = False
                    thread.last_touched_at = source_event.at
                    receipt.resolved_thread_ids.append(thread.thread_id)
                    allowed_resolutions += 1
                    break
            else:
                receipt.rejected.append(f"thread_missing:{review.thread_id}")

        if proposal.active_intent:
            intent = str(proposal.active_intent).strip()[: self.policy.max_intent_chars]
            next_state.working.active_intent = intent or None
            receipt.active_intent = intent or None

        next_state.rest.cycles += 1
        next_state.rest.last_cycle_at = source_event.at
        next_state.rest.last_cycle_source_count = len(source_memories)
        next_state.rest.last_cycle_new_memories = len(new_memories)
        next_state.rest_mode = False
        next_state.updated_at = source_event.at
        next_state.last_event_at = source_event.at
        next_state.normalize()
        return next_state, new_memories, links, receipt
