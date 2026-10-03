from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.factorized_boundary import (
    FactorizedBoundaryConfig,
    FactorizedBoundaryModule,
)
from nolane_personal.seeded_sampling import (
    SAMPLER_SCHEMA,
    SeededNucleusSampler,
    sample_sequence,
    splitmix64_next,
)
from nolane_personal.standalone_model import StandaloneNolaneLM


def small_model() -> StandaloneNolaneLM:
    torch.manual_seed(701)
    boundary = FactorizedBoundaryModule(
        FactorizedBoundaryConfig(
            vocab_size=37,
            hidden_size=12,
            rank=5,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex = DeepRecurrentStateSpaceCortex(
        12,
        DeepRecurrentCortexConfig(
            latent_dim=4,
            state_dim=3,
            virtual_steps=2,
            max_virtual_steps=3,
        ),
        seed=709,
    )
    return StandaloneNolaneLM(
        boundary,
        cortex,
        [0.1, -0.3, 0.2, 0.4],
    ).eval()


def test_splitmix64_known_answer_vectors():
    assert splitmix64_next(0) == (
        0x9E3779B97F4A7C15,
        0xE220A8397B1DCDAF,
    )
    assert splitmix64_next(1) == (
        0x9E3779B97F4A7C16,
        0x910A2DEC89025CC1,
    )
    assert splitmix64_next(0x123456789ABCDEF0) == (
        0xB06BD0321A075B05,
        0x161922C645CE50E8,
    )


def test_splitmix64_is_repeatable_and_stateful():
    first_state, first_value = splitmix64_next(0x123456789ABCDEF0)
    second_state, second_value = splitmix64_next(first_state)

    again_state, again_value = splitmix64_next(0x123456789ABCDEF0)
    assert (first_state, first_value) == (again_state, again_value)
    assert second_state != first_state
    assert second_value != first_value


def test_seeded_sampler_is_repeatable_and_rejects_invalid_inputs():
    rows = [
        [0.1, 1.2, -0.4, 0.8],
        [1.0, 0.9, 0.8, 0.7],
        [-1.0, -0.5, -0.2, -0.1],
        [0.0, 0.0, 0.0, 0.0],
    ]
    left, left_state = sample_sequence(rows, seed=123456789)
    right, right_state = sample_sequence(rows, seed=123456789)
    assert left == right
    assert left_state == right_state
    assert all(0 <= token < 4 for token in left)

    with pytest.raises(ValueError, match="non-finite"):
        SeededNucleusSampler(1).sample([0.0, math.nan])
    with pytest.raises(ValueError, match="top_p"):
        SeededNucleusSampler(1, top_p=0.0)
    with pytest.raises(ValueError, match="temperature"):
        SeededNucleusSampler(1, temperature=0.0)
    with pytest.raises(ValueError, match="quantization range"):
        SeededNucleusSampler(1).sample([1e20, 0.0])


def test_top_p_retains_minimal_q32_nucleus_with_stable_tie_break():
    logits = [0.0, 0.0, 0.0, 0.0]

    quarter = SeededNucleusSampler(
        state=123,
        temperature=1.0,
        top_p=0.25,
    )
    before = quarter.state
    token = quarter.sample(logits)
    expected_state, _ = splitmix64_next(before)
    assert token == 0
    assert quarter.state == expected_state

    for seed in range(32):
        half = SeededNucleusSampler(
            state=seed,
            temperature=1.0,
            top_p=0.50,
        )
        assert half.sample(logits) in {0, 1}

    observed = {
        SeededNucleusSampler(
            state=seed,
            temperature=1.0,
            top_p=1.0,
        ).sample(logits)
        for seed in range(64)
    }
    assert observed.issubset({0, 1, 2, 3})
    assert len(observed) > 2


def test_tiny_positive_top_p_still_retains_exactly_one_best_token():
    logits = [8.0, 1.0, 0.0, -4.0]
    for seed in range(16):
        sampler = SeededNucleusSampler(
            state=seed,
            temperature=0.78,
            top_p=1e-9,
        )
        assert sampler.sample(logits) == 0


def test_standalone_seeded_generation_is_exactly_repeatable():
    model = small_model()
    prompt = torch.tensor([[1, 5, 9, 7]], dtype=torch.long)
    kwargs = dict(
        input_ids=prompt,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.78,
        top_p=0.9,
        sampling_seed=0xC0FFEE1234567890,
        eos_token_id=None,
        pad_token_id=0,
    )

    first = model.generate(**kwargs)
    second = model.generate(**kwargs)
    assert torch.equal(first, second)
    assert first.shape == (1, prompt.shape[1] + 8)


def test_standalone_seeded_generation_changes_with_seed():
    model = small_model()
    prompt = torch.tensor([[1, 5, 9, 7]], dtype=torch.long)
    common = dict(
        input_ids=prompt,
        max_new_tokens=16,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=0,
    )
    first = model.generate(
        **common,
        sampling_seed=111,
    )
    second = model.generate(
        **common,
        sampling_seed=222,
    )
    assert not torch.equal(first, second)


def test_sampler_schema_is_frozen():
    assert SAMPLER_SCHEMA == "NOLANE-V054-SEEDED-Q32-NUCLEUS-V1"
