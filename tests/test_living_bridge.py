import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.bridge_cortex import LivingBridgeCortexConfig, TrainableLivingBridgeCortex
from nolane_personal.bridge_artifact import load_trained_bridge
from nolane_personal.bridge_training import (
    BridgeTrainingConfig,
    mean_encoded_nll,
    save_trained_bridge,
    train_encoded_examples,
)
from nolane_personal.living_bridge import (
    CrossLayerLivingBridge,
    LivingBridgeConfig,
    LivingBridgeHookSession,
    analytical_bridge_parameter_count,
)
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen():
    return Qwen3ForCausalLM(
        Qwen3Config(
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
    )


def _example(tokens, latent=0.35):
    ids = list(tokens)
    labels = [-100] + [3] * (len(ids) - 1)
    return (ids, labels, [latent] * 32, 1.0)


def test_bridge_exact_parameter_count_and_real_qwen_depth_trace():
    torch.manual_seed(5)
    model = _tiny_qwen().eval()
    config = LivingBridgeConfig(bridge_dim=32, max_layers=64)
    bridge = CrossLayerLivingBridge(32, config, seed=8)
    assert bridge.parameter_count() == analytical_bridge_parameter_count(32, config)
    assert bridge.parameter_count() == 17857
    assert analytical_bridge_parameter_count(1024, config) == 84321

    inputs = {"input_ids": torch.tensor([[1, 5, 7, 9]], dtype=torch.long)}
    with LivingBridgeHookSession(
        model,
        bridge,
        [0.2] * 32,
        layer_indices=[0, 1, 2],
        token_scope="all",
    ) as session:
        out = model(**inputs)

    assert out.logits.shape == (1, 4, 43)
    assert [row["layer_index"] for row in session.trace] == [0, 1, 2]
    assert all(abs(row["gate_mean"]) <= config.max_abs_gate + 1e-6 for row in session.trace)
    assert all(row["residual_norm"] > 0 for row in session.trace)
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)


def test_bridge_path_is_latent_sensitive_on_same_prompt():
    torch.manual_seed(17)
    model = _tiny_qwen().eval()
    bridge = CrossLayerLivingBridge(32, seed=3)
    cortex = TrainableLivingBridgeCortex(
        model,
        bridge,
        [0.8] * 32,
        config=LivingBridgeCortexConfig(layer_indices=(1, 2), token_scope="all"),
    )
    ids = torch.tensor([[1, 6, 8, 10]], dtype=torch.long)
    with torch.no_grad():
        positive = cortex.forward(input_ids=ids).logits.detach().clone()
        cortex.set_latent([-0.8] * 32)
        negative = cortex.forward(input_ids=ids).logits.detach().clone()
    assert not torch.allclose(positive, negative)


def test_real_qwen3_bridge_training_changes_only_bridge_and_improves_heldout():
    torch.manual_seed(23)
    model = _tiny_qwen().eval()
    bridge = CrossLayerLivingBridge(
        32,
        LivingBridgeConfig(
            latent_dim=32,
            bridge_dim=32,
            max_layers=64,
            max_abs_gate=0.25,
            initial_gate=0.08,
        ),
        seed=29,
    )
    cortex = TrainableLivingBridgeCortex(
        model,
        bridge,
        [0.35] * 32,
        config=LivingBridgeCortexConfig(layer_indices=(0, 1, 2), token_scope="all"),
    )
    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    heldout = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]

    baseline = mean_encoded_nll(cortex, heldout, personalized=False)
    before = module_parameter_digest(bridge.module)
    receipt = train_encoded_examples(
        cortex,
        train,
        config=BridgeTrainingConfig(epochs=35, learning_rate=0.025, max_grad_norm=1.0),
    )
    after = mean_encoded_nll(cortex, heldout, personalized=True)

    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.bridge_gradients_seen > 0
    assert receipt.bridge_changed
    assert module_parameter_digest(bridge.module) != before
    assert receipt.final_loss < receipt.initial_loss
    assert after < baseline
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)


def test_bridge_generation_uses_recurrent_path_and_cleans_hooks():
    torch.manual_seed(31)
    model = _tiny_qwen().eval()
    bridge = CrossLayerLivingBridge(32, seed=11)
    cortex = TrainableLivingBridgeCortex(
        model,
        bridge,
        [0.15] * 32,
        config=LivingBridgeCortexConfig(layer_indices=(1, 2), token_scope="all"),
    )
    output = cortex.generate(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert output.shape[1] == 5
    assert cortex.last_trace
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)



def test_bridge_artifact_roundtrip_and_lineage_fail_closed(tmp_path):
    torch.manual_seed(41)
    model = _tiny_qwen().eval()
    bridge = CrossLayerLivingBridge(32, seed=12)
    cortex = TrainableLivingBridgeCortex(
        model,
        bridge,
        [0.25] * 32,
        config=LivingBridgeCortexConfig(layer_indices=(1, 2), token_scope="all"),
    )
    receipt = train_encoded_examples(
        cortex,
        [_example([1, 5, 6, 7, 8, 9])],
        config=BridgeTrainingConfig(epochs=2, learning_rate=0.01),
    )
    manifest = save_trained_bridge(
        tmp_path,
        cortex,
        receipt,
        base_model_fingerprint="tiny-qwen3",
        dataset_fingerprint="protocol-sha",
    )
    loaded, config, meta = load_trained_bridge(
        tmp_path / "living-bridge.pt",
        expected_base_model_fingerprint="tiny-qwen3",
        expected_hidden_size=32,
    )
    assert loaded.parameter_count() == bridge.parameter_count()
    assert config.layer_indices == (1, 2)
    assert meta["bridge_state_digest"] == manifest["bridge_state_digest"]
    assert meta["dataset_fingerprint"] == "protocol-sha"

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_trained_bridge(
            tmp_path / "living-bridge.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
    with pytest.raises(ValueError, match="hidden-size mismatch"):
        load_trained_bridge(
            tmp_path / "living-bridge.pt",
            expected_base_model_fingerprint="tiny-qwen3",
            expected_hidden_size=64,
        )
