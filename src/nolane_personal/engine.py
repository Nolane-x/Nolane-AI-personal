from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone

from .consolidation import ConsolidationReceipt, ConsolidationValidator
from .cortex import Cortex, CortexReply, CortexRequest, NullCortex
from .dynamics import advance_time, apply_event, seconds_between
from .events import LivingEvent
from .initiative import InitiativeDecision, InitiativeEngine
from .memory import MemoryRecord, rank_memories
from .mutations import MutationReceipt, MutationValidator
from .observer import SocialObserver
from .rest import DeterministicRestObserver, RestObserver, RestPolicy, RestScheduler
from .state import LivingState, OpenThread
from .store import LivingStore


@dataclass(slots=True)
class EngineResult:
    state: LivingState
    speech: str = ""
    initiative: InitiativeDecision | None = None
    observer_receipt: MutationReceipt | None = None
    observer_error: str | None = None
    rest_receipt: ConsolidationReceipt | None = None
    rest_error: str | None = None


class LivingEngine:
    """Event-driven persistent runtime. Language models remain bounded organs."""

    def __init__(
        self,
        store: LivingStore,
        *,
        cortex: Cortex | None = None,
        initiative: InitiativeEngine | None = None,
        observer: SocialObserver | None = None,
        mutation_validator: MutationValidator | None = None,
        rest_observer: RestObserver | None = None,
        rest_scheduler: RestScheduler | None = None,
        consolidation_validator: ConsolidationValidator | None = None,
        enable_rest: bool = True,
        memory_enabled: bool = True,
    ) -> None:
        self.store = store
        self.cortex = cortex or NullCortex()
        self.initiative = initiative or InitiativeEngine()
        self.observer = observer
        self.mutation_validator = mutation_validator or MutationValidator()
        self.rest_scheduler = rest_scheduler or RestScheduler()
        self.rest_observer = rest_observer or DeterministicRestObserver(self.rest_scheduler.policy)
        self.consolidation_validator = consolidation_validator or ConsolidationValidator()
        self.enable_rest = bool(enable_rest)
        self.memory_enabled = bool(memory_enabled)
        state = self.store.load_state()
        self.state = state if state is not None else self.store.initialize(LivingState())

    def _advance_to(self, at: str) -> None:
        dt = seconds_between(self.state.last_event_at or self.state.updated_at, at)
        self.state = advance_time(deepcopy(self.state), dt)

    def _relevant_memories(self, query: str, limit: int = 6) -> list[MemoryRecord]:
        if not self.memory_enabled:
            return []
        return rank_memories(self.store.memories(limit=300), query, limit=limit)

    def _rest_source_memories(self) -> list[MemoryRecord]:
        policy: RestPolicy = self.rest_scheduler.policy
        already_consolidated = self.store.consolidated_parent_ids()
        return [
            memory
            for memory in self.store.memories(limit=policy.memory_window)
            if memory.memory_id not in already_consolidated
        ]

    def _run_observer(self, text: str, source_event: LivingEvent) -> tuple[MutationReceipt | None, str | None]:
        if self.observer is None:
            return None, None
        try:
            proposal = self.observer.observe(text, deepcopy(self.state), source_event)
            next_state, memories, receipt = self.mutation_validator.apply(self.state, proposal, source_event)
            mutation_event = LivingEvent(
                kind="observer_mutation",
                payload={
                    "source_event_id": source_event.event_id,
                    "proposal": proposal.to_dict(),
                    "receipt": receipt.to_dict(),
                },
                source="mutation_validator",
                at=source_event.at,
                salience=0.25,
            )
            self.state = self.store.commit_transition(mutation_event, next_state, memories)
            return receipt, None
        except Exception as exc:
            error_name = type(exc).__name__
            rejected = LivingEvent(
                kind="observer_rejected",
                payload={"source_event_id": source_event.event_id, "error_type": error_name},
                source="mutation_validator",
                at=source_event.at,
                salience=0.0,
            )
            self.state.last_event_at = source_event.at
            self.state.updated_at = source_event.at
            self.state = self.store.commit_transition(rejected, deepcopy(self.state))
            return None, error_name

    def _run_rest_cycle(self, at: str) -> tuple[ConsolidationReceipt | None, str | None]:
        source_memories = self._rest_source_memories()
        rest_event = LivingEvent(
            kind="rest_cycle",
            payload={"candidate_memory_ids": [m.memory_id for m in source_memories]},
            source="rest_scheduler",
            at=at,
            salience=0.0,
        )
        try:
            rest_state = deepcopy(self.state)
            rest_state.rest_mode = True
            proposal = self.rest_observer.consolidate(source_memories, rest_state)
            source_map = {memory.memory_id: memory for memory in source_memories}
            next_state, memories, links, receipt = self.consolidation_validator.apply(
                rest_state,
                proposal,
                rest_event,
                source_map,
            )
            rest_event.payload["proposal"] = proposal.to_dict()
            rest_event.payload["receipt"] = receipt.to_dict()
            self.state = self.store.commit_transition(rest_event, next_state, memories, links)
            return receipt, None
        except Exception as exc:
            error_name = type(exc).__name__
            rejected = LivingEvent(
                kind="rest_rejected",
                payload={"error_type": error_name},
                source="consolidation_validator",
                at=at,
                salience=0.0,
            )
            failed = deepcopy(self.state)
            failed.rest_mode = False
            failed.rest.cycles += 1
            failed.rest.last_cycle_at = at
            failed.rest.last_cycle_source_count = len(source_memories)
            failed.rest.last_cycle_new_memories = 0
            failed.updated_at = at
            failed.last_event_at = at
            self.state = self.store.commit_transition(rejected, failed)
            return None, error_name

    def handle_user_message(self, text: str, *, at: str | None = None, reply: bool = True) -> EngineResult:
        event = LivingEvent(kind="user_message", payload={"text": text}, source="user", at=at or self._now())
        self._advance_to(event.at)
        self.state = apply_event(deepcopy(self.state), event)
        memory = MemoryRecord(
            text=text.strip(),
            kind="episodic",
            salience=max(0.25, min(0.9, 0.38 + 0.03 * min(len(text) / 40.0, 8.0))),
            source_event_id=event.event_id,
            created_at=event.at,
        )
        memories = [memory] if text.strip() and self.memory_enabled else []
        self.state = self.store.commit_transition(event, self.state, memories)

        receipt, observer_error = self._run_observer(text, event)

        speech = ""
        if reply:
            recent_messages = self.store.conversation_messages(limit=9)
            if (
                recent_messages
                and recent_messages[-1].get("role") == "user"
                and str(recent_messages[-1].get("text", "")).strip()
                == text.strip()
            ):
                recent_messages = recent_messages[:-1]
            recent_messages = [
                {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("text", "")),
                }
                for message in recent_messages[-8:]
                if message.get("role") in {"user", "assistant"}
                and str(message.get("text", "")).strip()
            ]
            relevant_memories = [
                memory
                for memory in self._relevant_memories(text)
                if memory.source_event_id != event.event_id
            ]
            reply_obj = self.cortex.generate(CortexRequest(
                mode="reply",
                intent="respond_to_user",
                user_text=text,
                state=deepcopy(self.state),
                memories=relevant_memories,
                recent_messages=recent_messages,
            ))
            speech = reply_obj.utterance.strip()
            if speech:
                self._record_speech(reply_obj, at=event.at)
        return EngineResult(
            deepcopy(self.state),
            speech=speech,
            observer_receipt=receipt,
            observer_error=observer_error,
        )

    def tick(self, *, at: str | None = None) -> EngineResult:
        at = at or self._now()
        event = LivingEvent(kind="clock_tick", payload={}, source="clock", at=at, salience=0.0)
        self._advance_to(event.at)
        self.state = apply_event(deepcopy(self.state), event)
        self.state = self.store.commit_transition(event, self.state)

        rest_receipt = None
        rest_error = None
        now = datetime.fromisoformat(at)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if self.enable_rest:
            due, _reasons = self.rest_scheduler.due(self.state, now=now)
            if due:
                rest_receipt, rest_error = self._run_rest_cycle(at)

        decision = self.initiative.decide(self.state, now=now)
        if not decision.speak:
            return EngineResult(
                deepcopy(self.state),
                initiative=decision,
                rest_receipt=rest_receipt,
                rest_error=rest_error,
            )

        query = decision.intent.replace("follow_up:", "")
        reply_obj = self.cortex.generate(CortexRequest(
            mode="initiative",
            intent=decision.intent,
            user_text=None,
            state=deepcopy(self.state),
            memories=self._relevant_memories(query),
        ))
        speech = reply_obj.utterance.strip()
        if speech:
            self._record_speech(reply_obj, at=at)
        return EngineResult(
            deepcopy(self.state),
            speech=speech,
            initiative=decision,
            rest_receipt=rest_receipt,
            rest_error=rest_error,
        )

    def run_rest_now(self, *, at: str | None = None) -> EngineResult:
        at = at or self._now()
        self._advance_to(at)
        receipt, error = self._run_rest_cycle(at)
        return EngineResult(deepcopy(self.state), rest_receipt=receipt, rest_error=error)

    def add_open_thread(self, topic: str, *, importance: float = 0.6, due_at: str | None = None, at: str | None = None) -> OpenThread:
        at = at or self._now()
        thread = OpenThread(
            thread_id=f"thread-{len(self.state.open_threads)+1}-{self.state.version+1}",
            topic=topic,
            importance=importance,
            created_at=at,
            last_touched_at=at,
            due_at=due_at,
        )
        event = LivingEvent(
            kind="thread_opened",
            payload={"thread_id": thread.thread_id, "topic": topic, "importance": importance, "due_at": due_at},
            source="cognition",
            at=at,
        )
        self._advance_to(at)
        self.state.open_threads.append(thread)
        self.state.last_event_at = at
        self.state.updated_at = at
        self.state = self.store.commit_transition(event, self.state)
        return thread

    def resolve_thread(self, thread_id: str, *, at: str | None = None) -> bool:
        at = at or self._now()
        found = False
        for thread in self.state.open_threads:
            if thread.thread_id == thread_id:
                thread.unresolved = False
                thread.last_touched_at = at
                found = True
                break
        if not found:
            return False
        event = LivingEvent(kind="thread_resolved", payload={"thread_id": thread_id}, source="cognition", at=at)
        self._advance_to(at)
        self.state.last_event_at = at
        self.state.updated_at = at
        self.state = self.store.commit_transition(event, self.state)
        return True

    def _record_speech(self, reply: CortexReply, *, at: str) -> None:
        event = LivingEvent(kind="assistant_speech", payload={"text": reply.utterance, "intent": reply.intent}, source="cortex", at=at)
        self.state = apply_event(deepcopy(self.state), event)
        self.state = self.store.commit_transition(event, self.state)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
