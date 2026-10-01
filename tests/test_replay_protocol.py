from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from nolane_personal.engine import LivingEngine
from nolane_personal.replay_protocol import build_replay_protocol, verify_replay_protocol
from nolane_personal.store import LivingStore


def test_chronological_replay_protocol_freezes_history_and_allows_append(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        engine.handle_user_message(f"m{i}", at=(base + timedelta(minutes=i)).isoformat(), reply=False)

    protocol = build_replay_protocol(store)
    assert protocol["counts"] == {"total": 10, "train": 7, "dev": 1, "test": 2}
    train_versions = [x["version"] for x in protocol["splits"]["train"]]
    test_versions = [x["version"] for x in protocol["splits"]["test"]]
    assert max(train_versions) < min(test_versions)
    verify_replay_protocol(store, protocol)

    engine.handle_user_message("new-after-freeze", at=(base + timedelta(hours=2)).isoformat(), reply=False)
    verify_replay_protocol(store, protocol)
    store.close()


def test_protocol_digest_tamper_is_rejected(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(6):
        engine.handle_user_message(f"m{i}", at=(base + timedelta(minutes=i)).isoformat(), reply=False)
    protocol = build_replay_protocol(store)
    tampered = deepcopy(protocol)
    tampered["splits"]["test"][0]["event_kind"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_replay_protocol(store, tampered)
    store.close()
