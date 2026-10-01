from nolane_personal.events import LivingEvent
from nolane_personal.mutations import MutationPolicy, MutationValidator
from nolane_personal.observer import MemoryProposal, SocialProposal, ThreadProposal
from nolane_personal.state import LivingState


def test_validator_clips_model_authority_and_records_provenance():
    state = LivingState()
    event = LivingEvent(kind="user_message", payload={"text": "hello"}, source="user")
    proposal = SocialProposal(
        affect_delta={"concern": 99.0, "unknown": 1.0},
        relationship_delta={"trust": 2.0},
        memories=[MemoryProposal(text="user definitely loves jazz", kind="fact", confidence=0.4)],
        open_threads=[ThreadProposal(topic="ask about exam", importance=2.0)],
        resolve_thread_ids=["missing"],
        intent="follow_up",
        uncertainty=3.0,
    )
    next_state, memories, receipt = MutationValidator(MutationPolicy()).apply(state, proposal, event)

    assert next_state.affect.concern == 0.12
    assert next_state.relationship.trust == 0.085
    assert memories[0].kind == "inference"
    assert memories[0].source_event_id == event.event_id
    assert memories[0].metadata["downgraded_from_fact"] is True
    assert receipt.downgraded_memories == 1
    assert receipt.uncertainty == 1.0
    assert "affect:unknown" in receipt.rejected
    assert "resolve_missing:missing" in receipt.rejected
    assert next_state.open_threads[0].importance == 1.0
