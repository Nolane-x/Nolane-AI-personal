from nolane_personal.state import LivingState


def test_schema_v1_state_upgrades_to_v2_with_rest_defaults():
    legacy = LivingState().to_dict()
    legacy["schema_version"] = 1
    legacy.pop("rest", None)
    restored = LivingState.from_dict(legacy)
    assert restored.schema_version == 2
    assert restored.rest.cycles == 0
    assert restored.rest.last_cycle_at is None
