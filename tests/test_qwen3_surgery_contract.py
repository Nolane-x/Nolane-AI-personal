import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM

from nolane_personal.surgery import (
    CounterfactualSurgeryProbe,
    LatentResidualAdapter,
    resolve_transformer_layers,
)


def test_real_qwen3_decoder_accepts_shadow_latent_adapter_without_weight_mutation():
    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).eval()
    adapter = LatentResidualAdapter(32, seed=17)
    inputs = {
        "input_ids": torch.tensor([[1, 5, 9, 12]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    layers = resolve_transformer_layers(model)
    assert len(layers) == 4

    probe = CounterfactualSurgeryProbe(
        model,
        adapter,
        base_model_fingerprint="tiny-qwen3-contract",
        latent_digest="latent-contract",
    )
    baseline_direct = model(**inputs).logits.detach().clone()
    served, receipt = probe.run(
        inputs,
        [0.2] * 32,
        layer_indices=[1, 2],
        gate=0.03,
        token_scope="last",
    )

    assert torch.allclose(served.logits, baseline_direct)
    assert receipt.base_model_unchanged
    assert receipt.mean_abs_logit_shift > 0.0
    assert receipt.layer_indices == [1, 2]
    assert receipt.authority == "COUNTERFACTUAL_ONLY_BASELINE_OUTPUT"
    assert all(len(layer._forward_hooks) == 0 for layer in layers)
