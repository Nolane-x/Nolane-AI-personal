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


def test_replay_records_can_resume_after_version(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    engine.handle_user_message("one", reply=False)
    engine.handle_user_message("two", reply=False)
    engine.handle_user_message("three", reply=False)
    first_page = store.replay_records(limit=2)
    assert [r["version"] for r in first_page] == [1, 2]
    resumed = store.replay_records(limit=2, after_version=2)
    assert [r["version"] for r in resumed] == [3]
    store.close()
