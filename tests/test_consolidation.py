from nolane_personal.consolidation import ConsolidationValidator
from nolane_personal.events import LivingEvent
from nolane_personal.memory import MemoryRecord
from nolane_personal.rest import ConsolidatedMemoryProposal, RestProposal, ThreadReviewProposal
from nolane_personal.state import LivingState, OpenThread


def _memory(text, *, kind="inference", confidence=0.8):
    return MemoryRecord(text=text, kind=kind, confidence=confidence, salience=0.6)


def test_rest_rejects_unknown_sources_and_downgrades_unsupported_fact():
    a = _memory("User may like jazz", kind="inference", confidence=0.8)
    b = _memory("User mentioned jazz again", kind="episodic", confidence=1.0)
    source = {a.memory_id: a, b.memory_id: b}
    event = LivingEvent(kind="rest_cycle", source="rest_scheduler")
    proposal = RestProposal(
        consolidated_memories=[
            ConsolidatedMemoryProposal(
                text="User likes jazz",
                kind="fact",
                confidence=0.99,
                salience=0.8,
                source_memory_ids=[a.memory_id, b.memory_id],
            ),
            ConsolidatedMemoryProposal(
                text="fabricated",
                kind="fact",
                confidence=1.0,
                salience=1.0,
                source_memory_ids=["does-not-exist", a.memory_id],
            ),
        ]
    )
    state, memories, links, receipt = ConsolidationValidator().apply(LivingState(), proposal, event, source)
    assert len(memories) == 1
    assert memories[0].kind == "inference"
    assert memories[0].confidence == 0.8
    assert receipt.downgraded_memories == 1
    assert "memory:unknown_source" in receipt.rejected
    assert len(links) == 2
    assert state.relationship.closeness == LivingState().relationship.closeness


def test_thread_resolution_requires_real_memory_evidence():
    evidence = _memory("The exam is finished", kind="episodic", confidence=1.0)
    thread = OpenThread(thread_id="exam", topic="ask how the exam went")
    state = LivingState(open_threads=[thread])
    event = LivingEvent(kind="rest_cycle", source="rest_scheduler")

    missing = RestProposal(
        thread_reviews=[ThreadReviewProposal(thread_id="exam", action="resolve", source_memory_ids=[])]
    )
    next_state, _m, _l, receipt = ConsolidationValidator().apply(
        state, missing, event, {evidence.memory_id: evidence}
    )
    assert next_state.open_threads[0].unresolved
    assert "thread_evidence_missing:exam" in receipt.rejected

    supported = RestProposal(
        thread_reviews=[
            ThreadReviewProposal(
                thread_id="exam",
                action="resolve",
                source_memory_ids=[evidence.memory_id],
            )
        ]
    )
    next_state, _m, _l, receipt = ConsolidationValidator().apply(
        state, supported, event, {evidence.memory_id: evidence}
    )
    assert not next_state.open_threads[0].unresolved
    assert receipt.resolved_thread_ids == ["exam"]
