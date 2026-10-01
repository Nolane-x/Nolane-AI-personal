import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from nolane_personal.progressive_artifact import (
    load_progressive_artifact,
    save_progressive_artifact,
)
from nolane_personal.progressive_cortex import ProgressiveReplacementCortex
from nolane_personal.progressive_replacement import (
    LayerSensitivity,
    ProgressivePlanConfig,
    build_progressive_plan,
    calibrate_layer_sensitivity,
    verify_progressive_plan,
)
from nolane_personal.progressive_training import (
    ProgressiveTrainingConfig,
    train_progressive_stages,
)
from nolane_personal.replacement_cortex import ReplacementCortexConfig
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen(layers=6):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=53,
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


def test_calibration_builds_monotonic_plan_and_protects_edge_layers():
    torch.manual_seed(5)
    model = _tiny_qwen(6).eval()
    rows = calibrate_layer_sensitivity(
        model,
        [[1, 5, 6, 7], [1, 8, 9, 10]],
        edge_layers_to_keep=1,
    )
    assert {row.layer_index for row in rows} == {1, 2, 3, 4}
    assert all(row.score >= 0 for row in rows)

    plan = build_progressive_plan(
        total_layers=6,
        sensitivity=rows,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
        config=ProgressivePlanConfig(
            target_fraction=0.5,
            max_selected_layers=4,
            edge_layers_to_keep=1,
        ),
    )
    verify_progressive_plan(
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    assert len(plan["target_layers"]) == 3
    assert 0 not in plan["target_layers"]
    assert 5 not in plan["target_layers"]
    stage_sets = [set(row["layer_indices"]) for row in plan["stages"]]
    assert all(a.issubset(b) for a, b in zip(stage_sets, stage_sets[1:]))
    assert stage_sets[-1] == set(plan["target_layers"])


def test_progressive_plan_rejects_lineage_and_digest_drift():
    sensitivity = [
        LayerSensitivity(layer_index=i, residual_rms_ratio=0.1+i*0.01, cosine_change=0.01, score=0.11+i*0.01)
        for i in range(1, 5)
    ]
    plan = build_progressive_plan(
        total_layers=6,
        sensitivity=sensitivity,
        base_model_fingerprint="base-a",
        dataset_fingerprint="data-a",
    )
    with pytest.raises(ValueError, match="base-model mismatch"):
        verify_progressive_plan(plan, base_model_fingerprint="base-b", dataset_fingerprint="data-a")

    tampered = json.loads(json.dumps(plan))
    tampered["target_layers"] = [1]
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_progressive_plan(tampered, base_model_fingerprint="base-a", dataset_fingerprint="data-a")


def test_cached_and_replay_safe_generation_match_and_state_policy_differs():
    torch.manual_seed(11)
    model = _tiny_qwen(6).eval()
    replacement = RecurrentBlockReplacement(
        32,
        BlockReplacementConfig(recurrent_dim=32, initial_gate=0.08),
        seed=13,
    )
    cortex = ProgressiveReplacementCortex(
        model,
        replacement,
        [0.2] * 32,
        config=ReplacementCortexConfig(layer_indices=(2,), carry_recurrent_state=False),
    )

    seen_cached = []
    original = replacement.replace

    def wrapped_cached(hidden, latent, *, layer_index, state=None):
        seen_cached.append(state is None)
        return original(hidden, latent, layer_index=layer_index, state=state)

    replacement.replace = wrapped_cached
    cached = cortex.generate_cached(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert seen_cached[0] is True
    assert any(flag is False for flag in seen_cached[1:])
    assert cortex.last_bypass_counts[2] >= 2

    seen_replay = []

    def wrapped_replay(hidden, latent, *, layer_index, state=None):
        seen_replay.append(state is None)
        return original(hidden, latent, layer_index=layer_index, state=state)

    replacement.replace = wrapped_replay
    replay = cortex.generate_replay_safe(
        input_ids=torch.tensor([[1, 5, 6]], dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    replacement.replace = original

    assert seen_replay
    assert all(seen_replay)
    assert torch.equal(cached, replay)


def test_progressive_stage_training_keeps_qwen_frozen_and_roundtrips_artifact(tmp_path):
    torch.manual_seed(17)
    model = _tiny_qwen(6).eval()
    replacement = RecurrentBlockReplacement(
        32,
        BlockReplacementConfig(recurrent_dim=32, initial_gate=0.08),
        seed=19,
    )
    plan = build_progressive_plan(
        total_layers=6,
        sensitivity=[
            LayerSensitivity(layer_index=2, residual_rms_ratio=0.10, cosine_change=0.01, score=0.11),
            LayerSensitivity(layer_index=3, residual_rms_ratio=0.11, cosine_change=0.01, score=0.12),
            LayerSensitivity(layer_index=1, residual_rms_ratio=0.12, cosine_change=0.01, score=0.13),
            LayerSensitivity(layer_index=4, residual_rms_ratio=0.13, cosine_change=0.01, score=0.14),
        ],
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
        config=ProgressivePlanConfig(
            target_fraction=0.5,
            max_selected_layers=3,
            edge_layers_to_keep=1,
        ),
    )
    first = tuple(plan["stages"][0]["layer_indices"])
    cortex = ProgressiveReplacementCortex(
        model,
        replacement,
        [0.3] * 32,
        config=ReplacementCortexConfig(layer_indices=first, carry_recurrent_state=False),
    )
    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    dev = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]
    before = module_parameter_digest(replacement.module)
    receipt = train_progressive_stages(
        cortex,
        train,
        dev,
        plan,
        config=ProgressiveTrainingConfig(
            distill_epochs_per_stage=1,
            task_epochs_per_stage=3,
            learning_rate=0.02,
            distill_learning_rate=0.015,
            max_dev_regression=100.0,
        ),
    )
    assert receipt.stages_accepted == len(plan["stages"])
    assert receipt.final_layer_indices == plan["target_layers"]
    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.replacement_changed
    assert module_parameter_digest(replacement.module) != before

    manifest = save_progressive_artifact(
        tmp_path,
        cortex,
        receipt,
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    loaded, config, meta = load_progressive_artifact(
        tmp_path / "progressive-block-replacement.pt",
        expected_base_model_fingerprint="tiny-qwen",
        expected_hidden_size=32,
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.parameter_count() == replacement.parameter_count()
    assert list(config.layer_indices) == plan["target_layers"]
    assert meta["plan_sha256"] == plan["plan_sha256"]
    assert meta["replacement_state_digest"] == manifest["replacement_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_progressive_artifact(
            tmp_path / "progressive-block-replacement.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
