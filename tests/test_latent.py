import json

import pytest

from nolane_personal.latent import LatentBindingError, LatentStore


def test_latent_persists_across_store_restart_with_digest(tmp_path):
    path = tmp_path / "latent.json"
    store = LatentStore(path)
    latent = store.initialize(
        identity_id="identity-1",
        checkpoint_sha256="checkpoint-a",
        latent_dim=4,
        protocol_sha256="protocol-a",
    )
    latent.values = [0.1, -0.2, 0.3, -0.4]
    latent.source_state_version = 9
    latent.sequence = 7
    store.save(latent)

    reopened = LatentStore(path)
    restored = reopened.load_bound(
        identity_id="identity-1",
        checkpoint_sha256="checkpoint-a",
        latent_dim=4,
        protocol_sha256="protocol-a",
    )
    assert restored is not None
    assert restored.values == [0.1, -0.2, 0.3, -0.4]
    assert restored.source_state_version == 9
    assert restored.sequence == 7
    assert restored.digest


def test_latent_binding_refuses_silent_checkpoint_change(tmp_path):
    store = LatentStore(tmp_path / "latent.json")
    store.initialize(
        identity_id="identity-1",
        checkpoint_sha256="checkpoint-a",
        latent_dim=2,
        protocol_sha256=None,
    )
    with pytest.raises(LatentBindingError, match="checkpoint mismatch"):
        store.load_bound(
            identity_id="identity-1",
            checkpoint_sha256="checkpoint-b",
            latent_dim=2,
            protocol_sha256=None,
        )


def test_latent_digest_tamper_is_rejected(tmp_path):
    path = tmp_path / "latent.json"
    store = LatentStore(path)
    store.initialize(
        identity_id="identity-1",
        checkpoint_sha256="checkpoint-a",
        latent_dim=2,
        protocol_sha256=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["values"][0] = 123.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load()
