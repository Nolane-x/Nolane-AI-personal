from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.factorized_boundary import (
    FactorizedBoundaryConfig,
    FactorizedBoundaryModule,
)
from nolane_personal.mobile_factorized import (
    PACKAGE_AUTHORITY,
    PACKAGE_SCHEMA,
    export_mobile_factorized_package,
    verify_mobile_factorized_package,
)
from nolane_personal.standalone_model import StandaloneNolaneLM


def model() -> StandaloneNolaneLM:
    torch.manual_seed(41)
    boundary = FactorizedBoundaryModule(
        FactorizedBoundaryConfig(
            vocab_size=31,
            hidden_size=12,
            rank=5,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex = DeepRecurrentStateSpaceCortex(
        12,
        DeepRecurrentCortexConfig(
            latent_dim=4,
            state_dim=3,
            virtual_steps=2,
            max_virtual_steps=3,
        ),
        seed=43,
    )
    return StandaloneNolaneLM(
        boundary,
        cortex,
        [0.1, -0.2, 0.3, -0.4],
    ).eval()


def test_mobile_package_roundtrip_is_privacy_preserving_and_bound(tmp_path):
    source_sha = "a" * 64
    package = tmp_path / "mobile"

    manifest = export_mobile_factorized_package(
        model(),
        package,
        source_checkpoint_sha256=source_sha,
    )
    verified = verify_mobile_factorized_package(
        package,
        expected_source_checkpoint_sha256=source_sha,
    )

    assert verified == manifest
    assert manifest["schema"] == PACKAGE_SCHEMA
    assert manifest["authority"] == PACKAGE_AUTHORITY
    assert manifest["source_checkpoint_sha256"] == source_sha
    assert manifest["tensor_count"] > 10
    assert manifest["privacy"] == {
        "contains_user_latent": False,
        "contains_tokenizer": False,
        "contains_chat_text": False,
        "contains_promotion_authority": False,
    }
    assert manifest["runtime"] == {
        "python_required": False,
        "pytorch_required": False,
        "autoregressive_loop_owned_by_native_host": True,
        "sampling_owned_by_native_host": True,
    }

    contract = json.loads(
        (package / "contract.json").read_text(encoding="utf-8")
    )
    assert contract["vocab_size"] == 31
    assert contract["latent_dim"] == 4
    assert contract["packed_state_dim"] == 9


def test_mobile_package_refuses_source_checkpoint_mismatch(tmp_path):
    package = tmp_path / "mobile"
    export_mobile_factorized_package(
        model(),
        package,
        source_checkpoint_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="source checkpoint mismatch"):
        verify_mobile_factorized_package(
            package,
            expected_source_checkpoint_sha256="c" * 64,
        )


def test_mobile_package_detects_contract_tamper(tmp_path):
    package = tmp_path / "mobile"
    export_mobile_factorized_package(
        model(),
        package,
        source_checkpoint_sha256="d" * 64,
    )
    contract = package / "contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["virtual_steps"] += 1
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        verify_mobile_factorized_package(package)


def test_mobile_package_detects_weights_tamper(tmp_path):
    package = tmp_path / "mobile"
    export_mobile_factorized_package(
        model(),
        package,
        source_checkpoint_sha256="e" * 64,
    )
    weights = package / "weights.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0x01
    weights.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="weights SHA-256 mismatch"):
        verify_mobile_factorized_package(package)


def test_mobile_package_refuses_nonempty_output(tmp_path):
    output = tmp_path / "mobile"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_mobile_factorized_package(
            model(),
            output,
            source_checkpoint_sha256="f" * 64,
        )
