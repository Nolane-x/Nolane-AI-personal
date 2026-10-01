from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from .memory import MemoryRecord, lexical_tokens
from .state import LivingState


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    value = datetime.fromisoformat(ts)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class RestPolicy:
    min_idle_seconds: float = 30 * 60.0
    min_cycle_interval_seconds: float = 45 * 60.0
    memory_window: int = 160
    max_source_memories: int = 8
    duplicate_similarity: float = 0.72
    minimum_cluster_size: int = 2
    max_new_memories: int = 4
    max_thread_reviews: int = 8


@dataclass(slots=True)
class ConsolidatedMemoryProposal:
    text: str
    kind: str
    confidence: float
    salience: float
    source_memory_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThreadReviewProposal:
    thread_id: str
    action: str = "keep"
    reason: str = ""


@dataclass(slots=True)
class RestProposal:
    consolidated_memories: list[ConsolidatedMemoryProposal] = field(default_factory=list)
    thread_reviews: list[ThreadReviewProposal] = field(default_factory=list)
    active_intent: str | None = None
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RestObserver(Protocol):
    def consolidate(self, memories: list[MemoryRecord], state: LivingState) -> RestProposal: ...


class DeterministicRestObserver:
    """Conservative no-LLM consolidation baseline.

    It only merges near-duplicate memories and never upgrades an inference into
    a fact. This provides a cheap always-available REST path and court baseline.
    """

    def __init__(self, policy: RestPolicy | None = None) -> None:
        self.policy = policy or RestPolicy()

    @staticmethod
    def _similarity(a: MemoryRecord, b: MemoryRecord) -> float:
        ta, tb = lexical_tokens(a.text), lexical_tokens(b.text)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def consolidate(self, memories: list[MemoryRecord], state: LivingState) -> RestProposal:
        candidates = memories[: self.policy.memory_window]
        used: set[str] = set()
        proposals: list[ConsolidatedMemoryProposal] = []

        for anchor in candidates:
            if anchor.memory_id in used:
                continue
            cluster = [anchor]
            for other in candidates:
                if other.memory_id == anchor.memory_id or other.memory_id in used:
                    continue
                if self._similarity(anchor, other) >= self.policy.duplicate_similarity:
                    cluster.append(other)
                if len(cluster) >= self.policy.max_source_memories:
                    break
            if len(cluster) < self.policy.minimum_cluster_size:
                continue

            ids = [m.memory_id for m in cluster]
            used.update(ids)
            best = max(cluster, key=lambda m: (m.confidence, m.salience, len(m.text)))
            kind = best.kind if best.kind in {"episodic", "preference", "inference", "fact"} else "inference"
            if kind == "fact" and any(m.kind != "fact" for m in cluster):
                kind = "inference"
            confidence = min(m.confidence for m in cluster)
            salience = min(1.0, max(m.salience for m in cluster) + 0.05)
            proposals.append(
                ConsolidatedMemoryProposal(
                    text=best.text,
                    kind=kind,
                    confidence=confidence,
                    salience=salience,
                    source_memory_ids=ids,
                    metadata={"strategy": "near_duplicate_merge", "cluster_size": len(cluster)},
                )
            )
            if len(proposals) >= self.policy.max_new_memories:
                break

        reviews = [
            ThreadReviewProposal(thread_id=t.thread_id, action="keep", reason="deterministic_baseline")
            for t in state.open_threads
            if t.unresolved
        ][: self.policy.max_thread_reviews]
        return RestProposal(consolidated_memories=proposals, thread_reviews=reviews)


class RestScheduler:
    def __init__(self, policy: RestPolicy | None = None) -> None:
        self.policy = policy or RestPolicy()

    def due(self, state: LivingState, now: datetime | None = None) -> tuple[bool, list[str]]:
        now = now or datetime.now(timezone.utc)
        last_user = _parse(state.last_user_event_at)
        last_rest = _parse(state.rest.last_cycle_at)
        reasons: list[str] = []

        if last_user is None:
            return False, ["no_user_history"]
        idle = (now - last_user).total_seconds()
        if idle < self.policy.min_idle_seconds:
            return False, ["user_not_idle_enough"]
        reasons.append("idle_window")

        if last_rest is not None:
            since_rest = (now - last_rest).total_seconds()
            if since_rest < self.policy.min_cycle_interval_seconds:
                return False, ["rest_cycle_cooldown"]
        return True, reasons
