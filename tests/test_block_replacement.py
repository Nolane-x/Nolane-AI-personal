import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.block_replacement import (
    BlockReplacementConfig,
    BlockReplacementSession,
    RecurrentBlockReplacement,
    analytical_replacement_parameter_count,
)
from nolane_personal.replacement_artifact import load_trained_replacement
from nolane_personal.replacement_cortex import ReplacementCortexConfig, RecurrentReplacementCortex
from nolane_personal.replacement_training import (
    ReplacementTrainingConfig,
    mean_encoded_nll,
    save_trained_replacement,
    train_encoded_examples,
)
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen():
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=47,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=False,
        )
    )


def _example(tokens, latent=0.3):
    ids = list(tokens)
    labels = [-100] + [3] * (len(ids) - 1)
    return (ids, labels, [latent] * 32, 1.0)


def test_exact_parameter_count_and_actual_qwen_block_bypass():
    torch.manual_seed(3)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(32, seed=7)
    assert replacement.parameter_count() == analytical_replacement_parameter_count(32)
    assert replacement.parameter_count() == 17825
    assert analytical_replacement_parameter_count(1024) == 84289

    layer = model.model.layers[1]
    original = layer.forward
    calls = {"original": 0}

    def counted(*args, **kwargs):
        calls["original"] += 1
        return original(*args, **kwargs)

    layer.forward = counted
    try:
        ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
        with BlockReplacementSession(
            model,
            replacement,
            [0.2] * 32,
            layer_indices=[1],
        ) as session:
            out = model(input_ids=ids, use_cache=False)
            assert out.logits.shape == (1, 4, 47)
            assert calls["original"] == 0
            assert session.bypass_counts[1] == 1
            assert 0.0 < session.gate_means[1][0] < 1.0
        assert layer.forward is counted
    finally:
        layer.forward = original


def test_replacement_session_restores_original_forward_on_exception():
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(32, seed=4)
    layer = model.model.layers[2]
    original = layer.forward

    def fail(*args, **kwargs):
        raise RuntimeError("replacement failure")

    replacement.replace = fail
    with pytest.raises(RuntimeError, match="replacement failure"):
        with BlockReplacementSession(model, replacement, [0.1] * 32, layer_indices=[2]):
            model(input_ids=torch.tensor([[1, 2]], dtype=torch.long), use_cache=False)
    assert layer.forward == original


def test_teacher_distillation_plus_task_training_uses_zero_qwen_gradients_and_improves_heldout(tmp_path):
    torch.manual_seed(17)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(
        32,
        BlockReplacementConfig(recurrent_dim=32, max_layers=64, initial_gate=0.08),
        seed=19,
    )
    cortex = RecurrentReplacementCortex(
        model,
        replacement,
        [0.3] * 32,
        config=ReplacementCortexConfig(layer_indices=(1,), carry_recurrent_state=False),
    )
    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    heldout = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]

    baseline = mean_encoded_nll(cortex, heldout, replacement_enabled=False)
    before = module_parameter_digest(replacement.module)
    receipt = train_encoded_examples(
        cortex,
        train,
        config=ReplacementTrainingConfig(
            distill_epochs=3,
            task_epochs=35,
            learning_rate=0.025,
            distill_learning_rate=0.02,
            max_grad_norm=1.0,
        ),
    )
    after = mean_encoded_nll(cortex, heldout, replacement_enabled=True)

    assert receipt.teacher_pairs_seen >= 6
    assert receipt.distill_initial_loss is not None
    assert receipt.distill_final_loss is not None
    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.replacement_gradients_seen > 0
    assert receipt.replacement_changed
    assert module_parameter_digest(replacement.module) != before
    assert receipt.task_final_loss < receipt.task_initial_loss
    assert after < baseline

    manifest = save_trained_replacement(
        tmp_path,
        cortex,
        receipt,
        base_model_fingerprint="tiny-qwen3",
        dataset_fingerprint="protocol-sha",
    )
    loaded, config, metadata = load_trained_replacement(
        tmp_path / "recurrent-block-replacement.pt",
        expected_base_model_fingerprint="tiny-qwen3",
        expected_hidden_size=32,
    )
    assert loaded.parameter_count() == replacement.parameter_count()
    assert config.layer_indices == (1,)
    assert metadata["replacement_state_digest"] == manifest["replacement_state_digest"]
    assert metadata["dataset_fingerprint"] == "protocol-sha"


def test_replacement_generation_bypasses_selected_block():
    torch.manual_seed(29)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(32, seed=31)
    cortex = RecurrentReplacementCortex(
        model,
        replacement,
        [0.2] * 32,
        config=ReplacementCortexConfig(layer_indices=(1,), carry_recurrent_state=False),
    )
    output = cortex.generate(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert output.shape[1] == 5
    assert cortex.last_bypass_counts[1] >= 1
