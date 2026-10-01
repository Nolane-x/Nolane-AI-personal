import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.island_plan import IslandSensitivity
from nolane_personal.multiscale_cortex import (
    MultiTimescaleCortexConfig,
    MultiTimescaleStateSpaceCortex,
    analytical_multiscale_parameter_count,
)
from nolane_personal.scaffold_artifact import (
    load_scaffold_artifact,
    save_scaffold_artifact,
)
from nolane_personal.scaffold_plan import (
    ScaffoldPlanConfig,
    build_scaffold_plan,
    calibrate_scaffold_regions,
    verify_scaffold_plan,
)
from nolane_personal.scaffold_training import train_scaffold_stages
from nolane_personal.state_space_model import (
    StateSpaceModelConfig,
    StateSpacePersonalModel,
)
from nolane_personal.state_space_region import CortexRegion, StateSpaceRegionSession
from nolane_personal.state_space_training import StateSpaceTrainingConfig
from nolane_personal.surgery import module_parameter_digest


def _tiny_qwen(layers=12):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=67,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
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
    return build_scaffold_plan(
        total_layers=12,
        sensitivity=[
            _sensitivity(3, 8, 0.01),
            _sensitivity(2, 7, 0.20),
            _sensitivity(4, 9, 0.30),
            _sensitivity(2, 9, 0.02),
        ],
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
        config=ScaffoldPlanConfig(
            target_remaining_fraction=1/3,
            stage_remaining_fractions=(0.50, 1/3),
            min_head_layers=2,
            min_tail_layers=2,
        ),
    )


def test_multiscale_exact_parameter_count_and_scan_equivalence():
    torch.manual_seed(3)
    config = MultiTimescaleCortexConfig(state_dim=24, slow_decay_floor=0.50)
    cortex = MultiTimescaleStateSpaceCortex(32, config, seed=7)
    assert cortex.parameter_count() == analytical_multiscale_parameter_count(32, config)
    assert cortex.parameter_count() == 6849
    assert analytical_multiscale_parameter_count(1024, config) == 81249

    hidden = torch.randn(1, 7, 32)
    latent = [0.25] * 32
    full, full_state, trace = cortex.scan(hidden, latent, state=None)

    state = None
    pieces = []
    for index in range(hidden.shape[1]):
        piece, state, _ = cortex.scan(
            hidden[:, index:index+1, :],
            latent,
            state=state,
        )
        pieces.append(piece)
    incremental = torch.cat(pieces, dim=1)

    assert torch.allclose(full, incremental, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_state, state, atol=1e-6, rtol=1e-6)
    assert trace["tokens_scanned"] == 7
    assert trace["mean_slow_decay"] >= config.slow_decay_floor

    d = config.state_dim
    assert not torch.allclose(
        full_state[:, :d],
        full_state[:, d:],
        atol=1e-6,
        rtol=1e-6,
    )


def test_multiscale_cortex_preserves_living_latent_direction():
    torch.manual_seed(5)
    cortex = MultiTimescaleStateSpaceCortex(32, seed=11)
    hidden = torch.randn(1, 4, 32)
    positive, _, _ = cortex.scan(hidden, [0.8] * 32)
    negative, _, _ = cortex.scan(hidden, [-0.8] * 32)
    assert not torch.allclose(positive, negative)


def test_thin_qwen_shell_executes_only_outer_blocks():
    torch.manual_seed(13)
    qwen = _tiny_qwen().eval()
    cortex = MultiTimescaleStateSpaceCortex(32, seed=17)
    original_calls = {index: 0 for index in range(12)}
    originals = {}

    for index in original_calls:
        layer = qwen.model.layers[index]
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
            qwen,
            cortex,
            [0.2] * 32,
            region=CortexRegion(2, 9),
        ) as session:
            out = qwen(
                input_ids=torch.tensor([[1, 5, 7, 9]], dtype=torch.long),
                use_cache=False,
            )
            assert out.logits.shape == (1, 4, 67)
            assert scan_calls["count"] == 1
            assert session.cortex_calls == 1
            assert all(original_calls[index] == 0 for index in range(2, 10))
            assert all(original_calls[index] == 1 for index in (0, 1, 10, 11))
            assert sum(session.identity_calls.values()) == 7
    finally:
        cortex.scan = original_scan
        for index, original in originals.items():
            qwen.model.layers[index].forward = original


def test_real_qwen_calibration_and_scaffold_plan_shrink_monotonically():
    torch.manual_seed(19)
    model = _tiny_qwen().eval()
    config = ScaffoldPlanConfig(
        target_remaining_fraction=1/3,
        stage_remaining_fractions=(0.50, 1/3),
        min_head_layers=2,
        min_tail_layers=2,
    )
    rows = calibrate_scaffold_regions(
        model,
        [[1, 5, 6, 7], [1, 8, 9, 10]],
        config=config,
    )
    assert rows
    assert min(row.start for row in rows) >= 2
    assert max(row.end for row in rows) <= 9

    plan = _two_stage_plan()
    verify_scaffold_plan(
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    assert plan["stages"][0]["remaining_qwen_layers"] == 6
    assert plan["stages"][0]["region"] == {"start": 3, "end": 8}
    assert plan["target"]["remaining_qwen_layers"] == 4
    assert plan["target"]["region"] == {"start": 2, "end": 9}
    assert plan["target"]["replaced_layers"] == 8

    tampered = json.loads(json.dumps(plan))
    tampered["target"]["remaining_qwen_layers"] = 6
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_scaffold_plan(
            tampered,
            base_model_fingerprint="tiny-qwen",
            dataset_fingerprint="protocol",
        )


def test_cached_and_replay_safe_multiscale_generation_match():
    torch.manual_seed(23)
    qwen = _tiny_qwen().eval()
    cortex = MultiTimescaleStateSpaceCortex(
        32,
        MultiTimescaleCortexConfig(state_dim=24, initial_gate=0.08),
        seed=29,
    )
    model = StateSpacePersonalModel(
        qwen,
        cortex,
        [0.2] * 32,
        config=StateSpaceModelConfig(
            region=CortexRegion(2, 9),
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


def test_scaffold_training_shrinks_qwen_and_roundtrips_artifact(tmp_path):
    torch.manual_seed(31)
    qwen = _tiny_qwen().eval()
    cortex = MultiTimescaleStateSpaceCortex(
        32,
        MultiTimescaleCortexConfig(state_dim=24, initial_gate=0.08),
        seed=37,
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
    receipt = train_scaffold_stages(
        model,
        train,
        dev,
        plan,
        config=StateSpaceTrainingConfig(
            distill_epochs_per_stage=1,
            task_epochs_per_stage=2,
            learning_rate=0.02,
            distill_learning_rate=0.015,
            max_grad_norm=1.0,
            max_dev_regression=100.0,
        ),
    )
    assert receipt.stages_accepted == 2
    assert receipt.final_region == {"start": 2, "end": 9}
    assert receipt.head_layers == 2
    assert receipt.tail_layers == 2
    assert receipt.remaining_qwen_layers == 4
    assert receipt.replaced_layers == 8
    assert receipt.base_model_unchanged
    assert receipt.base_gradients_seen == 0
    assert receipt.cortex_gradients_seen > 0
    assert receipt.cortex_changed
    assert module_parameter_digest(cortex.module) != before

    manifest = save_scaffold_artifact(
        tmp_path,
        model,
        receipt,
        plan,
        base_model_fingerprint="tiny-qwen",
        dataset_fingerprint="protocol",
    )
    loaded, config, metadata = load_scaffold_artifact(
        tmp_path / "shrinking-qwen-scaffold.pt",
        expected_base_model_fingerprint="tiny-qwen",
        expected_hidden_size=32,
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.parameter_count() == cortex.parameter_count()
    assert config.region == CortexRegion(2, 9)
    assert metadata["plan_sha256"] == plan["plan_sha256"]
    assert metadata["cortex_state_digest"] == manifest["cortex_state_digest"]

    with pytest.raises(ValueError, match="base-model mismatch"):
        load_scaffold_artifact(
            tmp_path / "shrinking-qwen-scaffold.pt",
            expected_base_model_fingerprint="wrong",
            expected_hidden_size=32,
        )
