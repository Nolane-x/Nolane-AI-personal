from nolane_personal.engine import LivingEngine
from nolane_personal.observer import MemoryProposal, SocialProposal
from nolane_personal.store import LivingStore


class Observer:
    def observe(self, text, state, source_event):
        return SocialProposal(
            affect_delta={"playfulness": 0.4},
            relationship_delta={"closeness": 0.2},
            memories=[MemoryProposal(text="possible preference", kind="preference", confidence=0.7)],
            intent="continue_topic",
            uncertainty=0.2,
        )


class BrokenObserver:
    def observe(self, text, state, source_event):
        raise ValueError("bad model output")


def test_engine_commits_validated_observer_transition(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store, observer=Observer())
    result = engine.handle_user_message("hello", reply=False)
    assert result.observer_receipt is not None
    assert result.observer_error is None
    assert engine.state.affect.playfulness <= 0.57
    assert engine.state.relationship.closeness <= 0.089
    assert any(m.text == "possible preference" for m in store.memories())
    assert len(store.replay_records()) == 2
    store.close()


def test_observer_failure_is_fail_closed_and_audited(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store, observer=BrokenObserver())
    before = engine.state.relationship.trust
    result = engine.handle_user_message("hello", reply=False)
    assert result.observer_receipt is None
    assert result.observer_error == "ValueError"
    assert engine.state.relationship.trust == before
    assert len(store.replay_records()) == 2
    store.close()
