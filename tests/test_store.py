from nolane_personal.dynamics import apply_event
from nolane_personal.events import LivingEvent
from nolane_personal.memory import MemoryRecord
from nolane_personal.state import LivingState
from nolane_personal.store import LivingStore


def test_store_survives_restart_and_has_digest(tmp_path):
    db = tmp_path / "living.db"
    store = LivingStore(db)
    state = store.initialize(LivingState())
    identity = state.identity_id
    event = LivingEvent(kind="user_message", payload={"text": "hello"}, source="user")
    state = apply_event(state, event)
    state = store.commit_transition(event, state, [MemoryRecord(text="hello", source_event_id=event.event_id)])
    digest = store.snapshot_digest()
    version = state.version
    store.close()

    reopened = LivingStore(db)
    restored = reopened.load_state()
    assert restored is not None
    assert restored.identity_id == identity
    assert restored.version == version
    assert reopened.snapshot_digest() == digest
    assert reopened.memories()[0].text == "hello"
    reopened.close()


def test_rollback_creates_new_auditable_transition(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    initial = store.initialize(LivingState())
    event = LivingEvent(kind="user_message", payload={"text": "hi"}, source="user")
    changed = apply_event(initial, event)
    changed = store.commit_transition(event, changed)
    rolled = store.rollback(0)
    assert rolled.identity_id == initial.identity_id
    assert rolled.version == changed.version + 1
    assert rolled.relationship.interaction_count == 0
