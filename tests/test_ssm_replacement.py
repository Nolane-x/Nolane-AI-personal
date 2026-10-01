import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.ssm_distill import (
    SSMTrainingConfig,
    capture_hidden_pairs,
    distill_hidden_pairs,
)
from nolane_personal.ssm_replacement import (
    BlockReplacementSession,
    SSMReplacementConfig,
    SelectiveStateSpaceMixer,
    StateSpaceReplacementBlock,
    analytical_ssm_parameter_count,
)


def _tiny_qwen():
    config = Qwen3Config(
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
    return Qwen3ForCausalLM(config).eval()


def _batch(tokens):
    return {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
    }


def test_ssm_replacement_exact_parameter_count():
    config = SSMReplacementConfig(latent_dim=32, state_dim=8, max_abs_gate=1.0)
    mixer = SelectiveStateSpaceMixer(32, config, seed=1)
    assert analytical_ssm_parameter_count(32, config) == 889
    assert mixer.parameter_count() == 889


def test_real_qwen3_transformer_layer_is_actually_bypassed_and_restored():
    torch.manual_seed(5)
    model = _tiny_qwen()
    original = model.model.layers[2]
    calls = {"count": 0}
    original_forward = original.forward

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original_forward(*args, **kwargs)

    original.forward = counted
    mixer = SelectiveStateSpaceMixer(
        32,
        SSMReplacementConfig(state_dim=8),
        seed=2,
    )
    with torch.no_grad():
        mixer.raw_gate.fill_(0.3)
    replacement = StateSpaceReplacementBlock(mixer, [0.2] * 32)

    with BlockReplacementSession(model, 2, replacement):
        out = model(**_batch([1, 5, 6, 7]), use_cache=False)
        assert torch.isfinite(out.logits).all()
        assert calls["count"] == 0
        assert replacement.calls == 1
        assert model.model.layers[2] is replacement.module

    assert model.model.layers[2] is original
    model(**_batch([1, 5, 6, 7]), use_cache=False)
    assert calls["count"] == 1


def test_hidden_distillation_learns_original_qwen3_layer_mapping_and_improves_heldout():
    torch.manual_seed(17)
    model = _tiny_qwen()
    latent = [0.25] * 32
    train_batches = [
        _batch([1, 3, 4, 5, 6]),
        _batch([1, 7, 8, 9, 10]),
        _batch([1, 11, 12, 13, 14]),
    ]
    heldout_batches = [
        _batch([1, 15, 16, 17, 18]),
        _batch([1, 19, 20, 21, 22]),
    ]
    train_pairs = capture_hidden_pairs(
        model,
        2,
        train_batches,
        [latent] * len(train_batches),
    )
    heldout_pairs = capture_hidden_pairs(
        model,
        2,
        heldout_batches,
        [latent] * len(heldout_batches),
    )
    mixer = SelectiveStateSpaceMixer(
        32,
        SSMReplacementConfig(state_dim=16, max_abs_gate=1.0),
        seed=3,
    )

    def mse(pairs):
        mixer.eval()
        rows = []
        with torch.no_grad():
            for pair in pairs:
                pred, _ = mixer.scan(pair.input_hidden, pair.latent)
                rows.append(float(torch.nn.functional.mse_loss(pred, pair.target_hidden)))
        return sum(rows) / len(rows)

    heldout_before = mse(heldout_pairs)
    receipt = distill_hidden_pairs(
        mixer,
        train_pairs,
        config=SSMTrainingConfig(
            epochs=50,
            learning_rate=0.02,
            max_grad_norm=1.0,
            initial_effective_gate=0.35,
        ),
    )
    heldout_after = mse(heldout_pairs)

    assert receipt.gradients_seen > 0
    assert receipt.final_mse < receipt.initial_mse
    assert heldout_after < heldout_before


def test_replacement_changes_full_model_path_without_mutating_original_layer_weights():
    torch.manual_seed(23)
    model = _tiny_qwen()
    original = model.model.layers[1]
    before = {
        name: tensor.detach().clone()
        for name, tensor in original.state_dict().items()
    }
    batches = [
        _batch([1, 5, 7, 9]),
        _batch([1, 6, 8, 10]),
    ]
    latent = [0.1] * 32
    pairs = capture_hidden_pairs(model, 1, batches, [latent] * 2)
    mixer = SelectiveStateSpaceMixer(32, SSMReplacementConfig(state_dim=16), seed=6)
    distill_hidden_pairs(
        mixer,
        pairs,
        config=SSMTrainingConfig(epochs=20, learning_rate=0.02, initial_effective_gate=0.3),
    )
    replacement = StateSpaceReplacementBlock(mixer, latent)

    baseline = model(**_batch([1, 12, 13, 14]), use_cache=False).logits.detach()
    replacement.reset_state()
    with BlockReplacementSession(model, 1, replacement):
        candidate = model(**_batch([1, 12, 13, 14]), use_cache=False).logits.detach()

    assert torch.isfinite(candidate).all()
    assert not torch.allclose(baseline, candidate)
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, before[name])
