from datetime import datetime, timedelta, timezone

from nolane_personal.cortex import CortexReply
from nolane_personal.engine import LivingEngine
from nolane_personal.initiative import InitiativeEngine, InitiativePolicy
from nolane_personal.store import LivingStore


class FakeCortex:
    def generate(self, request):
        if request.mode == "initiative":
            return CortexReply("How did the math exam go?", request.intent)
        return CortexReply("Got it.", request.intent)


def test_engine_persists_identity_and_reply(tmp_path):
    db = tmp_path / "living.db"
    store = LivingStore(db)
    engine = LivingEngine(store, cortex=FakeCortex())
    identity = engine.state.identity_id
    result = engine.handle_user_message("I have a math exam tomorrow")
    assert result.speech == "Got it."
    assert engine.state.relationship.interaction_count == 1
    store.close()

    store2 = LivingStore(db)
    engine2 = LivingEngine(store2, cortex=FakeCortex())
    assert engine2.state.identity_id == identity
    assert engine2.state.relationship.interaction_count == 1
    store2.close()


def test_tick_can_initiate_without_new_user_prompt(tmp_path):
    now = datetime.now(timezone.utc)
    store = LivingStore(tmp_path / "living.db")
    policy = InitiativePolicy(threshold=0.45, min_user_silence_seconds=1, speech_cooldown_seconds=1)
    engine = LivingEngine(store, cortex=FakeCortex(), initiative=InitiativeEngine(policy))
    engine.handle_user_message("Tomorrow I have a math exam", at=(now - timedelta(hours=2)).isoformat(), reply=False)
    engine.add_open_thread("math exam", importance=1.0, at=(now - timedelta(hours=2)).isoformat())
    engine.state.affect.social_drive = 1.0
    engine.state.working.curiosity = 1.0
    result = engine.tick(at=now.isoformat())
    assert result.initiative is not None and result.initiative.speak
    assert result.speech == "How did the math exam go?"
    assert engine.state.last_ai_speech_at == now.isoformat()
