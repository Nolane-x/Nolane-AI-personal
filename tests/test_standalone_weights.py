import gc

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.native_artifact import (
    build_native_boundary_model,
    save_native_boundary_artifact,
)
from nolane_personal.native_boundary import (
    NativeBoundaryConfig,
    NativeNolaneBoundaryModel,
)
from nolane_personal.native_spec import build_native_boundary_spec
from nolane_personal.standalone_artifact import (
    export_standalone_from_l15,
    load_standalone_model,
)
from nolane_personal.standalone_model import clone_boundary_from_qwen
from nolane_personal.surgery import module_parameter_digest


class _Receipt:
    def to_dict(self):
        return {
            "schema": "TEST-L15-RECEIPT",
            "authority": "TEST_ONLY",
        }


def _tiny_qwen(*, tied=False):
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=73,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=96,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=True,
            tie_word_embeddings=tied,
        )
    )
    if tied:
        model.tie_weights()
    return model.eval()


def _make_l15(tmp_path, *, tied=False):
    torch.manual_seed(5)
    qwen = _tiny_qwen(tied=tied)
    cortex = DeepRecurrentStateSpaceCortex(
        32,
        DeepRecurrentCortexConfig(initial_gate=0.08),
        seed=7,
    )
    latent = [0.2] * 32
    model = NativeNolaneBoundaryModel(
        qwen,
        cortex,
        latent,
        config=NativeBoundaryConfig(carry_recurrent_state=False),
    )
    spec = build_native_boundary_spec(
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    l15_dir = tmp_path / "l15"
    manifest = save_native_boundary_artifact(
        l15_dir,
        model,
        _Receipt(),
        spec,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    return qwen, latent, l15_dir / "native-nolane-boundary.pt", manifest


def test_boundary_clone_owns_storage_and_preserves_tied_contract():
    qwen = _tiny_qwen(tied=False)
    boundary = clone_boundary_from_qwen(qwen)
    assert boundary.embed_tokens.weight.data_ptr() != qwen.model.embed_tokens.weight.data_ptr()
    assert boundary.lm_head.weight.data_ptr() != qwen.lm_head.weight.data_ptr()
    assert boundary.config.tie_word_embeddings is False

    tied = _tiny_qwen(tied=True)
    tied_boundary = clone_boundary_from_qwen(tied)
    assert tied_boundary.config.tie_word_embeddings is True
    assert tied_boundary.lm_head is None


def test_exported_standalone_matches_l15_without_qwen_runtime_reference(tmp_path):
    qwen, latent, l15_path, l15_manifest = _make_l15(tmp_path)
    out_dir = tmp_path / "l16"
    manifest = export_standalone_from_l15(
        qwen,
        l15_path,
        out_dir,
        latent=latent,
        expected_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )
    standalone, meta = load_standalone_model(
        out_dir / "standalone-nolane.pt",
        latent,
        device="cpu",
        expected_source_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )
    l15, l15_meta = build_native_boundary_model(
        qwen,
        l15_path,
        latent,
        expected_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )

    ids = torch.tensor([[1, 5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        source = l15.forward(input_ids=ids, state=None)
        owned = standalone.forward(input_ids=ids, state=None)
    assert torch.allclose(source.logits, owned.logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(source.state, owned.state, atol=1e-6, rtol=1e-6)
    assert standalone.prompt_scan_equivalent(ids)
    assert torch.equal(
        l15.generate(
            input_ids=ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
        ),
        standalone.generate(
            input_ids=ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
        ),
    )

    assert not hasattr(standalone, "qwen_model")
    assert meta["runtime_requires_qwen_model"] is False
    assert meta["runtime_requires_transformers"] is False
    assert meta["source_l15_checkpoint_sha256"] == l15_manifest["checkpoint_sha256"]
    assert meta["cortex_state_digest"] == l15_meta["cortex_state_digest"]
    assert manifest["runtime_requires_qwen_model"] is False
    assert manifest["runtime_requires_transformers"] is False


def test_standalone_isolated_from_source_mutation_and_survives_source_deletion(tmp_path):
    qwen, latent, l15_path, _manifest = _make_l15(tmp_path)
    out_dir = tmp_path / "l16"
    export_standalone_from_l15(
        qwen,
        l15_path,
        out_dir,
        latent=latent,
        expected_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )
    standalone, _ = load_standalone_model(
        out_dir / "standalone-nolane.pt",
        latent,
        device="cpu",
    )
    ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        before = standalone.forward(input_ids=ids).logits.detach().clone()
        qwen.model.embed_tokens.weight[5].add_(10.0)
        qwen.model.norm.weight.mul_(0.0)
        qwen.lm_head.weight[3].add_(10.0)
        after = standalone.forward(input_ids=ids).logits.detach().clone()
    assert torch.equal(before, after)

    del qwen
    gc.collect()
    generated = standalone.generate(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=None,
    )
    assert generated.shape[1] == 5


def test_standalone_checkpoint_contains_no_decoder_tensor_sections(tmp_path):
    qwen, latent, l15_path, _manifest = _make_l15(tmp_path)
    out_dir = tmp_path / "l16"
    export_standalone_from_l15(
        qwen,
        l15_path,
        out_dir,
        latent=latent,
        expected_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )
    payload = torch.load(
        out_dir / "standalone-nolane.pt",
        map_location="cpu",
        weights_only=True,
    )
    keys = list(payload["boundary_state"].keys()) + list(payload["cortex_state"].keys())
    assert not any("layers." in key for key in keys)
    assert not any("self_attn" in key or ".mlp." in key for key in keys)
    assert payload["runtime_requires_qwen_model"] is False
    assert payload["runtime_requires_transformers"] is False


def test_standalone_artifact_preserves_boundary_and_cortex_digests(tmp_path):
    qwen, latent, l15_path, _manifest = _make_l15(tmp_path)
    out_dir = tmp_path / "l16"
    manifest = export_standalone_from_l15(
        qwen,
        l15_path,
        out_dir,
        latent=latent,
        expected_base_model_fingerprint="tiny-qwen",
        expected_dataset_fingerprint="protocol",
    )
    model, meta = load_standalone_model(
        out_dir / "standalone-nolane.pt",
        latent,
        device="cpu",
    )
    assert module_parameter_digest(model.boundary.module) == manifest["boundary_state_digest"]
    assert module_parameter_digest(model.cortex.module) == manifest["cortex_state_digest"]
    assert meta["boundary_state_digest"] == manifest["boundary_state_digest"]
    assert meta["cortex_state_digest"] == manifest["cortex_state_digest"]
