import pytest

torch = pytest.importorskip("torch")

from nolane_personal.surgery import LatentAdapterConfig
from nolane_personal.surgery_candidate import (
    create_candidate,
    load_candidate,
    model_lock_fingerprint,
)


def _lock(revision="abc"):
    return {
        "upstream": {
            "provider": "huggingface",
            "repo_id": "Qwen/Qwen3-0.6B",
            "revision": revision,
            "architecture": "Qwen3ForCausalLM",
            "model_type": "qwen3",
        }
    }


def test_candidate_is_reproducible_and_bound_to_model_identity(tmp_path):
    lock = _lock()
    manifest = create_candidate(
        tmp_path,
        hidden_size=64,
        model_lock=lock,
        config=LatentAdapterConfig(latent_dim=32, bottleneck_dim=16, max_abs_gate=0.10),
        seed=42,
    )
    assert manifest["authority"] == "SHADOW_ONLY"
    assert manifest["parameter_count"] == 1681
    assert manifest["parameter_count"] == manifest["analytical_parameter_count"]

    adapter, metadata = load_candidate(
        tmp_path / "latent-adapter.pt",
        expected_model_lock_fingerprint=model_lock_fingerprint(lock),
        expected_hidden_size=64,
    )
    assert adapter.parameter_count() == 1681
    assert metadata["adapter_state_digest"] == manifest["adapter_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_candidate(
            tmp_path / "latent-adapter.pt",
            expected_model_lock_fingerprint=model_lock_fingerprint(_lock("different")),
            expected_hidden_size=64,
        )

    with pytest.raises(ValueError, match="hidden-size mismatch"):
        load_candidate(
            tmp_path / "latent-adapter.pt",
            expected_model_lock_fingerprint=model_lock_fingerprint(lock),
            expected_hidden_size=128,
        )
