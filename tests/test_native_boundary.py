import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.native_artifact import (
    load_native_boundary_artifact,
    save_native_boundary_artifact,
)
from nolane_personal.native_boundary import (
    NativeBoundaryConfig,
    NativeNolaneBoundaryModel,
)
from nolane_personal.native_court import run_with_decoder_call_count
from nolane_personal.native_spec import (
    build_native_boundary_spec,
    verify_native_boundary_spec,
)
from nolane_personal.native_training import (
    NativeTrainingConfig,
    train_native_boundary,
)
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen(layers=6):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=71,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=96,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=True,
        )
    )


def _example(tokens, latent=0.3):
    ids = list(tokens)
    labels = [-100] + [3] * (len(ids) - 1)
    return (ids, labels, [latent] * 32, 1.0)


def _native(seed=7):
    qwen = _tiny_qwen().eval()
    cortex = DeepRecurrentStateSpaceCortex(
        32,
        DeepRecurrentCortexConfig(initial_gate=0.08),
        seed=seed,
    )
    model = NativeNolaneBoundaryModel(
        qwen,
        cortex,
        [0.2] * 32,
        config=NativeBoundaryConfig(carry_recurrent_state=False),
    )
    return qwen, cortex, model


def test_native_forward_executes_zero_qwen_decoder_blocks():
    torch.manual_seed(3)
    qwen, _cortex, model = _native()
    ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    output, calls, per_layer = run_with_decoder_call_count(
        qwen,
        lambda: model.forward(input_ids=ids, state=None),
    )
    assert output.logits.shape == (1, 4, 71)
    assert calls == 0
    assert per_layer == [0] * 6


def test_native_generation_owns_state_and_executes_zero_qwen_decoder_blocks():
    torch.manual_seed(5)
    qwen, _cortex, model = _native(seed=11)
    ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    generated, calls, per_layer = run_with_decoder_call_count(
        qwen,
        lambda: model.generate(
            input_ids=ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
        ),
    )
    assert generated.shape[1] == 6
    assert calls == 0
    assert per_layer == [0] * 6
    assert model.prompt_scan_equivalent(ids)


def test_native_boundary_uses_only_embedding_norm_and_lm_head():
    torch.manual_seed(7)
    qwen, _cortex, model = _native(seed=13)
    ids = torch.tensor([[1, 8, 9]], dtype=torch.long)

    embed_calls = {"n": 0}
    norm_calls = {"n": 0}
    head_calls = {"n": 0}
    handles = [
        qwen.model.embed_tokens.register_forward_hook(
            lambda *_: embed_calls.__setitem__("n", embed_calls["n"] + 1)
        ),
        qwen.model.norm.register_forward_hook(
            lambda *_: norm_calls.__setitem__("n", norm_calls["n"] + 1)
        ),
        qwen.lm_head.register_forward_hook(
            lambda *_: head_calls.__setitem__("n", head_calls["n"] + 1)
        ),
    ]
    try:
        model.forward(input_ids=ids, state=None)
    finally:
        for handle in handles:
            handle.remove()
    assert embed_calls["n"] == 1
    assert norm_calls["n"] == 1
    assert head_calls["n"] == 1


def test_native_spec_is_frozen_and_fail_closed():
    spec = build_native_boundary_spec(
        base_model_fingerprint="base",
        dataset_fingerprint="protocol",
    )
    verify_native_boundary_spec(
        spec,
        base_model_fingerprint="base",
        dataset_fingerprint="protocol",
    )
    assert spec["qwen_decoder_layers_executed"] == 0
    assert spec["hf_kv_cache_used"] is False

    bad = dict(spec)
    bad["qwen_decoder_layers_executed"] = 1
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_native_boundary_spec(
            bad,
            base_model_fingerprint="base",
            dataset_fingerprint="protocol",
        )


def test_teacher_distillation_changes_only_native_cortex_and_roundtrips(tmp_path):
    torch.manual_seed(17)
    qwen, cortex, model = _native(seed=19)
    spec = build_native_boundary_spec(
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    before = module_parameter_digest(cortex.module)
    receipt = train_native_boundary(
        model,
        train,
        spec,
        config=NativeTrainingConfig(
            epochs=4,
            learning_rate=0.02,
            distill_weight=0.25,
            distill_temperature=2.0,
            max_grad_norm=1.0,
        ),
    )
    assert receipt.optimizer_steps == 8
    assert receipt.cortex_changed
    assert receipt.cortex_gradients_seen > 0
    assert receipt.qwen_model_unchanged
    assert receipt.qwen_gradients_seen == 0
    assert module_parameter_digest(cortex.module) != before

    manifest = save_native_boundary_artifact(
        tmp_path,
        model,
        receipt,
        spec,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    loaded, config, metadata = load_native_boundary_artifact(
        tmp_path / "native-nolane-boundary.pt",
        expected_base_model_fingerprint="tiny-qwen",
        expected_hidden_size=32,
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.parameter_count() == cortex.parameter_count()
    assert config.carry_recurrent_state is False
    assert metadata["spec_sha256"] == spec["spec_sha256"]
    assert metadata["cortex_state_digest"] == manifest["cortex_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_native_boundary_artifact(
            tmp_path / "native-nolane-boundary.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
