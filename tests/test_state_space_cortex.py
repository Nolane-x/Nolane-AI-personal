import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.island_plan import IslandSensitivity
from nolane_personal.state_space_artifact import (
    load_state_space_artifact,
    save_state_space_artifact,
)
from nolane_personal.state_space_core import (
    SelectiveStateSpaceCortex,
    StateSpaceCortexConfig,
    analytical_state_space_parameter_count,
)
from nolane_personal.state_space_model import (
    StateSpaceModelConfig,
    StateSpacePersonalModel,
)
from nolane_personal.state_space_plan import (
    StateSpacePlanConfig,
    build_state_space_plan,
    calibrate_state_space_regions,
    verify_state_space_plan,
)
from nolane_personal.state_space_region import (
    CortexRegion,
    StateSpaceRegionSession,
)
from nolane_personal.state_space_training import (
    StateSpaceTrainingConfig,
    train_state_space_stages,
)
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen(layers=8):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=61,
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


def _two_stage_plan():
    return build_state_space_plan(
        total_layers=8,
        sensitivity=[
            _sensitivity(2, 3, 0.01),
            _sensitivity(1, 4, 0.02),
            _sensitivity(3, 4, 0.30),
            _sensitivity(2, 5, 0.40),
        ],
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
        config=StateSpacePlanConfig(
            target_fraction=0.50,
            stage_fractions=(0.25, 0.50),
            edge_layers_to_keep=1,
            min_region_width=2,
            max_region_width=4,
        ),
    )


def test_state_space_exact_parameter_count_and_scan_equivalence():
    torch.manual_seed(3)
    config = StateSpaceCortexConfig(state_dim=32)
    cortex = SelectiveStateSpaceCortex(32, config, seed=7)
    assert cortex.parameter_count() == analytical_state_space_parameter_count(32, config)
    assert cortex.parameter_count() == 6433
    assert analytical_state_space_parameter_count(1024, config) == 72897

    hidden = torch.randn(1, 6, 32)
    latent = [0.25] * 32
    full, full_state, full_trace = cortex.scan(hidden, latent, state=None)

    pieces = []
    state = None
    for index in range(hidden.shape[1]):
        piece, state, _trace = cortex.scan(
            hidden[:, index:index+1, :],
            latent,
            state=state,
        )
        pieces.append(piece)
    incremental = torch.cat(pieces, dim=1)

    assert torch.allclose(full, incremental, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_state, state, atol=1e-6, rtol=1e-6)
    assert full_trace["tokens_scanned"] == 6


def test_state_space_preserves_latent_direction():
    torch.manual_seed(5)
    cortex = SelectiveStateSpaceCortex(32, seed=9)
    hidden = torch.randn(1, 4, 32)
    positive, _, _ = cortex.scan(hidden, [0.8] * 32)
    negative, _, _ = cortex.scan(hidden, [-0.8] * 32)
    assert not torch.allclose(positive, negative)


def test_one_state_space_call_replaces_wide_qwen_region():
    torch.manual_seed(11)
    model = _tiny_qwen().eval()
    cortex = SelectiveStateSpaceCortex(32, seed=13)
    original_calls = {2: 0, 3: 0, 4: 0, 5: 0}
    originals = {}
    for index in original_calls:
        layer = model.model.layers[index]
        originals[index] = layer.forward

        def counted(*args, _index=index, _original=layer.forward, **kwargs):
            original_calls[_index] += 1
            return _original(*args, **kwargs)

        layer.forward = counted

    scan_calls = {"count": 0}
    original_scan = cortex.scan

    def counted_scan(*args, **kwargs):
        scan_calls["count"] += 1
        return original_scan(*args, **kwargs)

    cortex.scan = counted_scan
    try:
        with StateSpaceRegionSession(
            model,
            cortex,
            [0.2] * 32,
            region=CortexRegion(2, 5),
        ) as session:
            output = model(
                input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
                use_cache=False,
            )
            assert output.logits.shape == (1, 4, 61)
            assert original_calls == {2: 0, 3: 0, 4: 0, 5: 0}
            assert scan_calls["count"] == 1
            assert session.cortex_calls == 1
            assert session.identity_calls == {3: 1, 4: 1, 5: 1}
    finally:
        cortex.scan = original_scan
        for index, original in originals.items():
            model.model.layers[index].forward = original


def test_real_qwen_calibration_builds_nested_widening_plan():
    torch.manual_seed(17)
    model = _tiny_qwen().eval()
    config = StateSpacePlanConfig(
        target_fraction=0.50,
        stage_fractions=(0.25, 0.50),
        edge_layers_to_keep=1,
        min_region_width=2,
        max_region_width=4,
    )
    rows = calibrate_state_space_regions(
        model,
        [[1, 5, 6, 7], [1, 8, 9, 10]],
        config=config,
    )
    assert rows
    assert {row.width for row in rows} >= {2, 4}

    plan = _two_stage_plan()
    verify_state_space_plan(
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    assert plan["stages"][0]["region"] == {"start": 2, "end": 3}
    assert plan["stages"][1]["region"] == {"start": 1, "end": 4}
    assert plan["target_width"] == 4

    tampered = json.loads(json.dumps(plan))
    tampered["target_width"] = 2
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_state_space_plan(
            tampered,
            base_model_fingerprint="tiny-qwen",
            dataset_fingerprint="protocol",
        )


def test_cached_and_replay_safe_state_space_generation_match():
    torch.manual_seed(19)
    qwen = _tiny_qwen().eval()
    cortex = SelectiveStateSpaceCortex(
        32,
        StateSpaceCortexConfig(state_dim=32, initial_gate=0.08),
        seed=23,
    )
    model = StateSpacePersonalModel(
        qwen,
        cortex,
        [0.2] * 32,
        config=StateSpaceModelConfig(
            region=CortexRegion(2, 5),
            carry_recurrent_state=False,
        ),
    )
    ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    cached = model.generate_cached(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    cached_calls = model.last_cortex_calls
    model.reset_state()
    replay = model.generate_replay_safe(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert torch.equal(cached, replay)
    assert cached_calls >= 2
    assert model.last_identity_calls[3] >= 2


def test_multistage_state_space_training_keeps_qwen_frozen_and_roundtrips(tmp_path):
    torch.manual_seed(29)
    qwen = _tiny_qwen().eval()
    cortex = SelectiveStateSpaceCortex(
        32,
        StateSpaceCortexConfig(state_dim=32, initial_gate=0.08),
        seed=31,
    )
    plan = _two_stage_plan()
    first = plan["stages"][0]["region"]
    model = StateSpacePersonalModel(
        qwen,
        cortex,
        [0.3] * 32,
        config=StateSpaceModelConfig(
            region=CortexRegion(first["start"], first["end"]),
            carry_recurrent_state=False,
        ),
    )
    train = [
        _example([1, 5, 6, 7, 8, 9]),
        _example([1, 10, 11, 12, 13, 14]),
    ]
    dev = [
        _example([1, 15, 16, 17, 18, 19]),
        _example([1, 20, 21, 22, 23, 24]),
    ]
    before = module_parameter_digest(cortex.module)
    receipt = train_state_space_stages(
        model,
        train,
        dev,
        plan,
        config=StateSpaceTrainingConfig(
            distill_epochs_per_stage=1,
            task_epochs_per_stage=3,
            learning_rate=0.02,
            distill_learning_rate=0.015,
            max_grad_norm=1.0,
            max_dev_regression=100.0,
        ),
    )
    assert receipt.stages_accepted == 2
    assert receipt.final_region == {"start": 1, "end": 4}
    assert receipt.replaced_layers == 4
    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.cortex_gradients_seen > 0
    assert receipt.cortex_changed
    assert module_parameter_digest(cortex.module) != before
    assert all(row["teacher_examples_seen"] > 0 for row in receipt.stage_receipts)

    manifest = save_state_space_artifact(
        tmp_path,
        model,
        receipt,
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    loaded, config, metadata = load_state_space_artifact(
        tmp_path / "selective-state-space-cortex.pt",
        expected_base_model_fingerprint="tiny-qwen",
        expected_hidden_size=32,
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.parameter_count() == cortex.parameter_count()
    assert config.region == CortexRegion(1, 4)
    assert metadata["plan_sha256"] == plan["plan_sha256"]
    assert metadata["cortex_state_digest"] == manifest["cortex_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_state_space_artifact(
            tmp_path / "selective-state-space-cortex.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
