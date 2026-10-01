from datetime import datetime, timedelta, timezone

from nolane_personal.engine import LivingEngine
from nolane_personal.store import LivingStore


def test_automatic_rest_consolidates_duplicates_with_append_only_provenance(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)

    engine.handle_user_message("Tôi thích nhạc jazz", at=base.isoformat(), reply=False)
    engine.handle_user_message("Tôi thích nhạc jazz", at=(base + timedelta(minutes=1)).isoformat(), reply=False)
    before_ids = {m.memory_id for m in store.memories()}

    result = engine.tick(at=(base + timedelta(minutes=40)).isoformat())
    assert result.rest_error is None
    assert result.rest_receipt is not None
    assert len(result.rest_receipt.stored_memory_ids) == 1
    assert result.rest_receipt.memory_links == 2
    assert engine.state.rest.cycles == 1
    assert not engine.state.rest_mode

    memories = store.memories()
    after_ids = {m.memory_id for m in memories}
    assert before_ids.issubset(after_ids)
    links = store.memory_links()
    assert len(links) == 2
    assert {link["parent_memory_id"] for link in links} == before_ids

    second = engine.tick(at=(base + timedelta(minutes=50)).isoformat())
    assert second.rest_receipt is None
    assert engine.state.rest.cycles == 1
    store.close()


def test_manual_rest_with_no_duplicate_evidence_is_audited_but_non_destructive(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    engine.handle_user_message("unique memory", reply=False)
    before = {m.memory_id for m in store.memories()}
    result = engine.run_rest_now()
    assert result.rest_error is None
    assert result.rest_receipt is not None
    assert result.rest_receipt.stored_memory_ids == []
    assert {m.memory_id for m in store.memories()} == before
    assert engine.state.rest.cycles == 1
    store.close()
