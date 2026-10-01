import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.personal_artifact import load_trained_adapter
from nolane_personal.personal_cortex import PersonalCortexConfig, TrainablePersonalCortex
from nolane_personal.personal_evaluation import mean_encoded_nll
from nolane_personal.personal_training import (
    PersonalTrainingConfig,
    save_trained_adapter,
    train_encoded_examples,
)
from nolane_personal.surgery import LatentAdapterConfig, LatentResidualAdapter, module_parameter_digest


def _tiny_qwen():
    config = Qwen3Config(
        vocab_size=41,
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
    return Qwen3ForCausalLM(config)


def _example(tokens):
    input_ids = list(tokens)
    labels = [-100] + [3] * (len(input_ids) - 1)
    return (input_ids, labels, [0.35] * 32, 1.0)


def test_real_qwen3_adapter_training_changes_only_personal_path_and_improves_heldout(tmp_path):
    torch.manual_seed(7)
    model = _tiny_qwen().eval()
    adapter = LatentResidualAdapter(
        32,
        LatentAdapterConfig(latent_dim=32, bottleneck_dim=16, max_abs_gate=0.25),
        seed=19,
    )
    personal = TrainablePersonalCortex(
        model,
        adapter,
        [0.35] * 32,
        config=PersonalCortexConfig(layer_indices=(1, 2), token_scope="all"),
    )

    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    heldout = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]

    baseline_heldout = mean_encoded_nll(personal, heldout, personalized=False)
    adapter_before = module_parameter_digest(adapter.module)
    receipt = train_encoded_examples(
        personal,
        train,
        config=PersonalTrainingConfig(
            epochs=35,
            learning_rate=0.03,
            max_grad_norm=1.0,
            initial_effective_gate=0.10,
            max_length=64,
        ),
    )
    personal_heldout = mean_encoded_nll(personal, heldout, personalized=True)

    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.adapter_gradients_seen > 0
    assert receipt.adapter_changed
    assert module_parameter_digest(adapter.module) != adapter_before
    assert receipt.final_loss < receipt.initial_loss
    assert personal_heldout < baseline_heldout

    manifest = save_trained_adapter(
        tmp_path,
        personal,
        receipt,
        base_model_fingerprint="tiny-qwen3",
        source_candidate_checkpoint_sha256="source-candidate",
        dataset_fingerprint="protocol-sha",
    )
    loaded, config, metadata = load_trained_adapter(
        tmp_path / "personal-cortex-adapter.pt",
        expected_base_model_fingerprint="tiny-qwen3",
        expected_hidden_size=32,
    )
    assert loaded.parameter_count() == adapter.parameter_count()
    assert config.layer_indices == (1, 2)
    assert metadata["adapter_state_digest"] == manifest["adapter_state_digest"]
    assert metadata["dataset_fingerprint"] == "protocol-sha"


def test_personal_generate_runs_real_qwen3_with_adapter_and_cleans_hooks():
    torch.manual_seed(13)
    model = _tiny_qwen().eval()
    adapter = LatentResidualAdapter(32, seed=4)
    with torch.no_grad():
        adapter.raw_gate.fill_(0.3)
    personal = TrainablePersonalCortex(
        model,
        adapter,
        [0.2] * 32,
        config=PersonalCortexConfig(layer_indices=(1, 2), token_scope="all"),
    )
    output = personal.generate(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert output.shape[1] == 5
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)
