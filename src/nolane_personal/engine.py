from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone

from .cortex import Cortex, CortexReply, CortexRequest, NullCortex
from .dynamics import advance_time, apply_event, seconds_between
from .events import LivingEvent
from .initiative import InitiativeDecision, InitiativeEngine
from .memory import MemoryRecord, rank_memories
from .mutations import MutationReceipt, MutationValidator
from .observer import SocialObserver
from .state import LivingState, OpenThread
from .store import LivingStore


@dataclass(slots=True)
class EngineResult:
    state: LivingState
    speech: str = ""
    initiative: InitiativeDecision | None = None
    observer_receipt: MutationReceipt | None = None
    observer_error: str | None = None


class LivingEngine:
    """Event-driven persistent runtime. The LLM is a replaceable cortex.

    Social observers have proposal authority only. All persistent observer
    mutations pass through MutationValidator and are written as separate,
    auditable transitions.
    """

    def __init__(
        self,
        store: LivingStore,
        *,
        cortex: Cortex | None = None,
        initiative: InitiativeEngine | None = None,
        observer: SocialObserver | None = None,
        mutation_validator: MutationValidator | None = None,
    ) -> None:
        self.store = store
        self.cortex = cortex or NullCortex()
        self.initiative = initiative or InitiativeEngine()
        self.observer = observer
        self.mutation_validator = mutation_validator or MutationValidator()
        state = self.store.load_state()
        self.state = state if state is not None else self.store.initialize(LivingState())

    def _advance_to(self, at: str) -> None:
        dt = seconds_between(self.state.last_event_at or self.state.updated_at, at)
        self.state = advance_time(deepcopy(self.state), dt)

    def _relevant_memories(self, query: str, limit: int = 6) -> list[MemoryRecord]:
        return rank_memories(self.store.memories(limit=300), query, limit=limit)

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
        memories = [memory] if text.strip() else []
        self.state = self.store.commit_transition(event, self.state, memories)

        receipt, observer_error = self._run_observer(text, event)

        speech = ""
        if reply:
            reply_obj = self.cortex.generate(CortexRequest(
                mode="reply",
                intent="respond_to_user",
                user_text=text,
                state=deepcopy(self.state),
                memories=self._relevant_memories(text),
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

        now = datetime.fromisoformat(at)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decision = self.initiative.decide(self.state, now=now)
        if not decision.speak:
            return EngineResult(deepcopy(self.state), initiative=decision)

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
        return EngineResult(deepcopy(self.state), speech=speech, initiative=decision)

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
