import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.hybrid_artifact import load_hybrid_artifact, save_hybrid_artifact
from nolane_personal.hybrid_training import HybridTrainingConfig, train_hybrid_encoded_examples
from nolane_personal.recurrent_cortex import (
    HybridCortexConfig,
    HybridRecurrentCortex,
    RecurrentCortexConfig,
    RecurrentCortexMixer,
    analytical_recurrent_parameter_count,
)
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen():
    config = Qwen3Config(
        vocab_size=43,
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


def _example(tokens):
    ids = list(tokens)
    labels = [-100] + [3] * (len(ids) - 1)
    return (ids, labels, [0.30] * 32, 1.0)


def _nll(cortex, examples):
    device = next(cortex.model.parameters()).device
    rows = []
    cortex.model.eval()
    cortex.mixer.eval()
    with torch.no_grad():
        for ids, labels, latent, weight in examples:
            cortex.set_latent(latent)
            cortex.reset_recurrent_state()
            output = cortex.forward(
                carry_state=False,
                input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                labels=torch.tensor([labels], dtype=torch.long, device=device),
                use_cache=False,
            )
            rows.append(float(output.loss.detach().cpu()) * weight)
    return sum(rows) / len(rows)


def test_recurrent_parameter_count_is_exact():
    config = RecurrentCortexConfig(latent_dim=32, recurrent_dim=8, max_abs_gate=0.10)
    mixer = RecurrentCortexMixer(32, config, seed=1)
    assert analytical_recurrent_parameter_count(32, config) == 2209
    assert mixer.parameter_count() == 2209


def test_recurrent_state_changes_same_input_and_reset_restores_trajectory():
    torch.manual_seed(3)
    model = _tiny_qwen()
    mixer = RecurrentCortexMixer(
        32,
        RecurrentCortexConfig(recurrent_dim=8, max_abs_gate=0.20),
        seed=4,
    )
    with torch.no_grad():
        mixer.raw_gate.fill_(0.7)
    cortex = HybridRecurrentCortex(
        model,
        mixer,
        [0.2] * 32,
        config=HybridCortexConfig(layer_indices=(1, 2), carry_across_calls=True),
    )
    ids = torch.tensor([[1, 5, 6, 7]], dtype=torch.long)

    first = cortex.forward(input_ids=ids, use_cache=False).logits.detach().clone()
    assert cortex.recurrent_states
    second = cortex.forward(input_ids=ids, use_cache=False).logits.detach().clone()
    assert not torch.allclose(first, second)

    cortex.reset_recurrent_state()
    replay = cortex.forward(input_ids=ids, use_cache=False).logits.detach().clone()
    assert torch.allclose(first, replay, atol=1e-6, rtol=1e-5)


def test_real_qwen3_recurrent_mixer_trains_with_zero_base_gradients_and_heldout_gain(tmp_path):
    torch.manual_seed(9)
    model = _tiny_qwen()
    mixer = RecurrentCortexMixer(
        32,
        RecurrentCortexConfig(recurrent_dim=12, max_abs_gate=0.25),
        seed=8,
    )
    cortex = HybridRecurrentCortex(
        model,
        mixer,
        [0.30] * 32,
        config=HybridCortexConfig(layer_indices=(1, 2), carry_across_calls=False),
    )

    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    heldout = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]
    baseline = _nll(cortex, heldout)
    before = module_parameter_digest(mixer.module)
    receipt = train_hybrid_encoded_examples(
        cortex,
        train,
        config=HybridTrainingConfig(
            epochs=35,
            learning_rate=0.025,
            max_grad_norm=1.0,
            initial_effective_gate=0.10,
        ),
    )
    personalized = _nll(cortex, heldout)

    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.mixer_gradients_seen > 0
    assert receipt.mixer_changed
    assert module_parameter_digest(mixer.module) != before
    assert receipt.final_loss < receipt.initial_loss
    assert personalized < baseline

    path = tmp_path / "hybrid.pt"
    manifest = save_hybrid_artifact(
        path,
        cortex,
        receipt,
        base_model_fingerprint="tiny-qwen3",
    )
    loaded, metadata = load_hybrid_artifact(
        path,
        expected_base_model_fingerprint="tiny-qwen3",
        expected_hidden_size=32,
        latent=[0.30] * 32,
        model=_tiny_qwen(),
    )
    assert loaded.mixer.parameter_count() == mixer.parameter_count()
    assert metadata["mixer_digest"] == manifest["mixer_digest"]


def test_recurrent_generate_keeps_state_across_decode_and_cleans_hooks():
    torch.manual_seed(12)
    model = _tiny_qwen()
    model.generation_config.use_cache = False
    mixer = RecurrentCortexMixer(32, RecurrentCortexConfig(recurrent_dim=8), seed=7)
    with torch.no_grad():
        mixer.raw_gate.fill_(0.4)
    cortex = HybridRecurrentCortex(
        model,
        mixer,
        [0.15] * 32,
        config=HybridCortexConfig(layer_indices=(1, 2)),
    )
    output = cortex.generate(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        use_cache=False,
        pad_token_id=0,
    )
    assert output.shape[1] == 5
    assert set(cortex.recurrent_states) == {1, 2}
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)
