from __future__ import annotations

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
from nolane_personal.mobile_factorized import (
    SCHEMA,
    build_mobile_factorized_modules,
    mobile_factorized_contract,
)
from nolane_personal.standalone_model import StandaloneNolaneLM


def mobile_model(*, tied: bool, seed: int = 17) -> StandaloneNolaneLM:
    torch.manual_seed(seed)
    boundary = FactorizedBoundaryModule(
        FactorizedBoundaryConfig(
            vocab_size=47,
            hidden_size=16,
            rank=7,
            rms_norm_eps=1e-6,
            tie_word_embeddings=tied,
            padding_idx=0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex = DeepRecurrentStateSpaceCortex(
        16,
        DeepRecurrentCortexConfig(
            latent_dim=6,
            state_dim=5,
            virtual_steps=3,
            max_virtual_steps=4,
            max_abs_gate=0.20,
            initial_gate=0.04,
            slow_decay_floor=0.55,
        ),
        seed=seed + 1,
    )
    return StandaloneNolaneLM(
        boundary,
        cortex,
        [0.15, -0.20, 0.05, 0.30, -0.10, 0.25],
    ).eval()


@pytest.mark.parametrize("tied", [False, True])
def test_mobile_init_state_matches_native_cortex(tied):
    model = mobile_model(tied=tied)
    init_state, _step, contract = build_mobile_factorized_modules(model)
    latent = torch.tensor([model.latent], dtype=torch.float32)

    with torch.no_grad():
        mobile = init_state(latent)
        native = model.cortex.initial_state(
            latent,
            device=latent.device,
            batch=1,
        )

    assert mobile.shape == (1, contract.packed_state_dim)
    assert torch.allclose(mobile, native, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("tied", [False, True])
def test_mobile_token_step_matches_native_incremental_logits_and_state(tied):
    model = mobile_model(tied=tied)
    init_state, step, contract = build_mobile_factorized_modules(model)
    latent = torch.tensor([model.latent], dtype=torch.float32)
    tokens = [1, 7, 11, 5, 19]

    with torch.no_grad():
        mobile_state = init_state(latent)
        native_state = None

        for token in tokens:
            token_id = torch.tensor([token], dtype=torch.long)
            mobile_logits, mobile_state = step(
                token_id,
                mobile_state,
                latent,
            )
            native = model.forward(
                input_ids=token_id.view(1, 1),
                state=native_state,
            )
            native_state = native.state

            assert mobile_logits.shape == (1, contract.vocab_size)
            assert mobile_state.shape == (1, contract.packed_state_dim)
            assert torch.allclose(
                mobile_logits,
                native.logits[:, -1, :],
                atol=1e-6,
                rtol=1e-6,
            )
            assert torch.allclose(
                mobile_state,
                native_state,
                atol=1e-6,
                rtol=1e-6,
            )


@pytest.mark.parametrize("tied", [False, True])
def test_mobile_modules_are_frozen_and_contract_is_explicit(tied):
    model = mobile_model(tied=tied)
    init_state, step, contract = build_mobile_factorized_modules(model)

    assert all(not parameter.requires_grad for parameter in init_state.parameters())
    assert all(not parameter.requires_grad for parameter in step.parameters())

    payload = contract.to_dict()
    assert payload["schema"] == SCHEMA
    assert payload["vocab_size"] == 47
    assert payload["hidden_size"] == 16
    assert payload["rank"] == 7
    assert payload["latent_dim"] == 6
    assert payload["state_dim"] == 5
    assert payload["packed_state_dim"] == 15
    assert payload["virtual_steps"] == 3
    assert payload["tie_word_embeddings"] is tied
    assert payload["hidden_norm_eps"] == pytest.approx(
        model.cortex.module.hidden_norm.eps
    )
    assert payload["latent_norm_eps"] == pytest.approx(
        model.cortex.module.latent_norm.eps
    )
    assert payload["input_contract"]["step_token_id"] == [1]
    assert payload["output_contract"]["step_logits"] == [1, 47]
    assert payload["autoregressive_loop_in_graph"] is False
    assert payload["sampling_in_graph"] is False
    assert payload["dynamic_eos_loop_in_graph"] is False


def test_mobile_contract_refuses_invalid_native_config_through_source_model():
    model = mobile_model(tied=True)
    contract = mobile_factorized_contract(model)
    assert contract.bos_token_id == 1
    assert contract.eos_token_id == 2
    assert contract.pad_token_id == 0
