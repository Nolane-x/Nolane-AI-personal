from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

torch = pytest.importorskip("torch")

from nolane_personal.engine import LivingEngine
from nolane_personal.living_core import LivingCoreConfig, TinyLivingCore
from nolane_personal.shadow import ShadowLivingCoreRunner
from nolane_personal.store import LivingStore


def test_shadow_core_persists_latent_and_resumes_only_new_transitions(tmp_path):
    db = tmp_path / "living.db"
    store = LivingStore(db)
    engine = LivingEngine(store)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        engine.handle_user_message(f"event-{i}", at=(base + timedelta(minutes=i)).isoformat(), reply=False)
    store.close()

    config = LivingCoreConfig()
    core = TinyLivingCore(config)
    checkpoint = tmp_path / "core.pt"
    torch.save(
        {
            "schema": "NOLANE-LIVING-CORE-CHECKPOINT-V2",
            "model_state": core.module.state_dict(),
            "config": asdict(config),
            "protocol_sha256": None,
            "return_horizon_seconds": 3600.0,
            "evidence_status": "TEST_ONLY",
        },
        checkpoint,
    )

    latent_path = tmp_path / "latent.json"
    runner = ShadowLivingCoreRunner(db, checkpoint, latent_path)
    first = runner.process_pending()
    assert first["processed"] == 3
    assert first["cursor_version"] == 3
    first_digest = first["latent_digest"]

    restarted = ShadowLivingCoreRunner(db, checkpoint, latent_path)
    second = restarted.process_pending()
    assert second["processed"] == 0
    assert second["cursor_version"] == 3
    assert second["latent_digest"] == first_digest

    store2 = LivingStore(db)
    engine2 = LivingEngine(store2)
    engine2.handle_user_message("event-3", at=(base + timedelta(minutes=3)).isoformat(), reply=False)
    store2.close()

    third = restarted.process_pending()
    assert third["processed"] == 1
    assert third["cursor_version"] == 4
    assert third["latent_sequence"] == 4
    assert third["authority"] == "SHADOW_ONLY_NO_STATE_MUTATION"
