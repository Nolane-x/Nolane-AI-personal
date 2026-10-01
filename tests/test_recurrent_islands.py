import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.block_replacement import BlockReplacementConfig, RecurrentBlockReplacement
from nolane_personal.island_artifact import load_island_artifact, save_island_artifact
from nolane_personal.island_cortex import IslandCortexConfig, RecurrentIslandCortex
from nolane_personal.island_plan import (
    IslandPlanConfig,
    IslandSensitivity,
    build_island_plan,
    calibrate_island_sensitivity,
    verify_island_plan,
)
from nolane_personal.island_replacement import (
    TransformerIsland,
    TransformerIslandSession,
    normalize_islands,
)
from nolane_personal.island_training import IslandTrainingConfig, train_island_stages
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen(layers=8):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=59,
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


def _sensitivity(start, end, score):
    width = end - start + 1
    return IslandSensitivity(
        start=start,
        end=end,
        width=width,
        residual_rms_ratio=score * width * 0.8,
        cosine_change=score * width * 0.2,
        transformation_score=score * width,
        score_per_layer=score,
    )


def _two_island_plan():
    return build_island_plan(
        total_layers=8,
        sensitivity=[
            _sensitivity(1, 2, 0.01),
            _sensitivity(4, 5, 0.02),
            _sensitivity(2, 3, 0.20),
            _sensitivity(5, 6, 0.30),
        ],
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
        config=IslandPlanConfig(
            target_fraction=0.5,
            min_width=2,
            max_width=2,
            max_islands=2,
            edge_layers_to_keep=1,
            min_gap_layers=1,
        ),
    )


def test_one_recurrent_call_replaces_entire_contiguous_qwen_region():
    torch.manual_seed(3)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(32, seed=7)
    original_calls = {2: 0, 3: 0, 4: 0}
    originals = {}

    for index in original_calls:
        layer = model.model.layers[index]
        originals[index] = layer.forward

        def counted(*args, _index=index, _original=layer.forward, **kwargs):
            original_calls[_index] += 1
            return _original(*args, **kwargs)

        layer.forward = counted

    replacement_calls = {"count": 0}
    original_replace = replacement.replace

    def counted_replace(*args, **kwargs):
        replacement_calls["count"] += 1
        return original_replace(*args, **kwargs)

    replacement.replace = counted_replace
    try:
        with TransformerIslandSession(
            model,
            replacement,
            [0.2] * 32,
            islands=[TransformerIsland(2, 4)],
        ) as session:
            output = model(input_ids=torch.tensor([[1, 5, 7, 9]]), use_cache=False)
            assert output.logits.shape == (1, 4, 59)
            assert original_calls == {2: 0, 3: 0, 4: 0}
            assert replacement_calls["count"] == 1
            assert session.island_calls[2] == 1
            assert session.identity_calls[3] == 1
            assert session.identity_calls[4] == 1
            assert session.replaced_layer_count == 3
            assert session.island_count == 1
    finally:
        replacement.replace = original_replace
        for index, original in originals.items():
            model.model.layers[index].forward = original


def test_islands_reject_overlap_and_enforce_gap():
    normalized = normalize_islands(
        [(1, 2), (4, 5)],
        total_layers=8,
        min_gap_layers=1,
    )
    assert normalized == (TransformerIsland(1, 2), TransformerIsland(4, 5))

    with pytest.raises(ValueError, match="overlap|minimum gap"):
        normalize_islands([(1, 3), (3, 4)], total_layers=8)

    with pytest.raises(ValueError, match="overlap|minimum gap"):
        normalize_islands([(1, 2), (3, 4)], total_layers=8, min_gap_layers=1)


def test_real_qwen_region_calibration_and_frozen_plan_contract():
    torch.manual_seed(5)
    model = _tiny_qwen().eval()
    config = IslandPlanConfig(
        target_fraction=0.35,
        min_width=2,
        max_width=3,
        max_islands=2,
        edge_layers_to_keep=1,
        min_gap_layers=1,
    )
    rows = calibrate_island_sensitivity(
        model,
        [[1, 5, 6, 7], [1, 8, 9, 10]],
        config=config,
    )
    assert rows
    assert all(row.width in {2, 3} for row in rows)
    assert all(row.start >= 1 and row.end <= 6 for row in rows)
    assert all(row.score_per_layer >= 0 for row in rows)

    plan = _two_island_plan()
    verify_island_plan(
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    assert plan["target_islands"] == [{"start": 1, "end": 2}, {"start": 4, "end": 5}]
    assert plan["replaced_layers"] == 4
    assert len(plan["stages"]) == 2

    tampered = json.loads(json.dumps(plan))
    tampered["target_islands"] = [{"start": 1, "end": 2}]
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_island_plan(
            tampered,
            base_model_fingerprint="tiny-qwen",
            dataset_fingerprint="protocol",
        )


def test_cached_and_replay_safe_island_generation_match():
    torch.manual_seed(11)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(
        32,
        BlockReplacementConfig(recurrent_dim=32, initial_gate=0.08),
        seed=13,
    )
    cortex = RecurrentIslandCortex(
        model,
        replacement,
        [0.2] * 32,
        config=IslandCortexConfig(
            islands=(TransformerIsland(2, 3),),
            carry_recurrent_state=False,
        ),
    )
    ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    cached = cortex.generate_cached(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    cached_calls = dict(cortex.last_island_calls)
    cortex.reset_state()
    replay = cortex.generate_replay_safe(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert torch.equal(cached, replay)
    assert cached_calls[2] >= 2
    assert cortex.last_identity_calls[3] >= 2


def test_multistage_region_distillation_keeps_qwen_frozen_and_roundtrips(tmp_path):
    torch.manual_seed(17)
    model = _tiny_qwen().eval()
    replacement = RecurrentBlockReplacement(
        32,
        BlockReplacementConfig(recurrent_dim=32, max_layers=64, initial_gate=0.08),
        seed=19,
    )
    plan = _two_island_plan()
    first = tuple(
        TransformerIsland(row["start"], row["end"])
        for row in plan["stages"][0]["islands"]
    )
    cortex = RecurrentIslandCortex(
        model,
        replacement,
        [0.3] * 32,
        config=IslandCortexConfig(islands=first, carry_recurrent_state=False),
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
    receipt = train_island_stages(
        cortex,
        train,
        dev,
        plan,
        config=IslandTrainingConfig(
            distill_epochs_per_stage=1,
            task_epochs_per_stage=3,
            learning_rate=0.02,
            distill_learning_rate=0.015,
            max_grad_norm=1.0,
            max_dev_regression=100.0,
        ),
    )
    assert receipt.stages_accepted == 2
    assert receipt.replaced_layers == 4
    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.replacement_gradients_seen > 0
    assert receipt.replacement_changed
    assert module_parameter_digest(replacement.module) != before
    assert all(row["teacher_regions_seen"] > 0 for row in receipt.stage_receipts)

    manifest = save_island_artifact(
        tmp_path,
        cortex,
        receipt,
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    loaded, config, metadata = load_island_artifact(
        tmp_path / "recurrent-transformer-islands.pt",
        expected_base_model_fingerprint="tiny-qwen",
        expected_hidden_size=32,
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.parameter_count() == replacement.parameter_count()
    assert config.islands == (TransformerIsland(1, 2), TransformerIsland(4, 5))
    assert metadata["plan_sha256"] == plan["plan_sha256"]
    assert metadata["replacement_state_digest"] == manifest["replacement_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_island_artifact(
            tmp_path / "recurrent-transformer-islands.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
