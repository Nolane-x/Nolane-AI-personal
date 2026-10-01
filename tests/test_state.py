from nolane_personal.state import LivingState, OpenThread


def test_state_roundtrip_preserves_identity_and_threads():
    state = LivingState()
    state.affect.valence = 3.0
    state.open_threads.append(OpenThread(thread_id="t1", topic="math exam", importance=0.8))
    restored = LivingState.from_dict(state.to_dict())
    assert restored.identity_id == state.identity_id
    assert restored.affect.valence == 1.0
    assert restored.open_threads[0].topic == "math exam"
