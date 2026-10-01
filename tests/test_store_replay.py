from nolane_personal.engine import LivingEngine
from nolane_personal.store import LivingStore


def test_replay_records_preserve_before_after_and_event(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    engine.handle_user_message("hello replay", reply=False)
    records = store.replay_records()
    assert len(records) == 1
    assert records[0]["event"].kind == "user_message"
    assert records[0]["before"].relationship.interaction_count == 0
    assert records[0]["after"].relationship.interaction_count == 1
    store.close()
