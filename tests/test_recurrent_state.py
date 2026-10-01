import json

import pytest

from nolane_personal.recurrent_state import RecurrentStateBundle, RecurrentStateStore


def _bundle():
    return RecurrentStateBundle(
        identity_id="id-1",
        base_model_fingerprint="base-a",
        mixer_digest="mixer-a",
        latent_digest="latent-a",
        recurrent_dim=4,
        layer_states={"1": [0.1, 0.2, 0.3, 0.4], "3": [-0.1, 0.0, 0.1, 0.2]},
        sequence=7,
    )


def test_recurrent_state_survives_restart_and_is_bound_to_model_mixer_and_latent(tmp_path):
    path = tmp_path / "recurrent.json"
    store = RecurrentStateStore(path)
    saved = store.save(_bundle())
    assert saved.digest

    reopened = RecurrentStateStore(path)
    loaded = reopened.load_bound(
        identity_id="id-1",
        base_model_fingerprint="base-a",
        mixer_digest="mixer-a",
        latent_digest="latent-a",
        recurrent_dim=4,
    )
    assert loaded is not None
    assert loaded.sequence == 7
    assert loaded.layer_states["1"] == [0.1, 0.2, 0.3, 0.4]

    with pytest.raises(ValueError, match="mixer_digest"):
        reopened.load_bound(
            identity_id="id-1",
            base_model_fingerprint="base-a",
            mixer_digest="mixer-b",
            latent_digest="latent-a",
            recurrent_dim=4,
        )


def test_recurrent_state_digest_detects_tamper(tmp_path):
    path = tmp_path / "recurrent.json"
    store = RecurrentStateStore(path)
    store.save(_bundle())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["layer_states"]["1"][0] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load()
