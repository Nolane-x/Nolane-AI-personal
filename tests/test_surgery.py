from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from nolane_personal.surgery import (
    CounterfactualSurgeryProbe,
    LatentAdapterConfig,
    LatentResidualAdapter,
    analytical_adapter_parameter_count,
    module_parameter_digest,
    resolve_transformer_layers,
)
from nolane_personal.surgery_court import evaluate_shadow_admission


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden):
        return (torch.tanh(self.proj(hidden)),)


class ToyBackbone(nn.Module):
    def __init__(self, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(hidden_size) for _ in range(layers)])


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 23, hidden_size: int = 16, layers: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden_size)
        self.model = ToyBackbone(hidden_size, layers)
        self.lm_head = nn.Linear(hidden_size, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_adapter_exact_parameter_count_and_gate_bound():
    config = LatentAdapterConfig(latent_dim=32, bottleneck_dim=16, max_abs_gate=0.10)
    adapter = LatentResidualAdapter(64, config, seed=7)
    assert analytical_adapter_parameter_count(64, config) == 1681
    assert adapter.parameter_count() == 1681
    assert float(adapter.effective_gate(999.0)) == pytest.approx(0.10)
    assert float(adapter.effective_gate(-999.0)) == pytest.approx(-0.10)


def test_parameter_digest_supports_bfloat16():
    module = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    digest = module_parameter_digest(module)
    assert len(digest) == 64


def test_counterfactual_probe_changes_shadow_logits_but_serves_baseline_and_preserves_model():
    torch.manual_seed(123)
    model = ToyLM()
    adapter = LatentResidualAdapter(16, seed=9)
    latent = [0.25] * 32
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}
    model_before = module_parameter_digest(model)
    baseline_direct = model(**inputs).logits.detach().clone()

    probe = CounterfactualSurgeryProbe(
        model,
        adapter,
        base_model_fingerprint="base-fingerprint",
        latent_digest="latent-digest",
    )
    served, receipt = probe.run(inputs, latent, layer_indices=[1, 2], gate=0.05)

    assert torch.allclose(served.logits, baseline_direct)
    assert receipt.authority == "COUNTERFACTUAL_ONLY_BASELINE_OUTPUT"
    assert receipt.base_model_unchanged
    assert module_parameter_digest(model) == model_before
    assert receipt.mean_abs_logit_shift > 0.0
    assert receipt.max_abs_logit_shift > 0.0
    assert receipt.adapter_parameters == analytical_adapter_parameter_count(16)
    assert receipt.layer_indices == [1, 2]
    assert all(len(layer._forward_hooks) == 0 for layer in resolve_transformer_layers(model))

    decision = evaluate_shadow_admission([receipt])
    assert decision.status == "SHADOW_ADMISSION_PASS"
    assert decision.summary["authority"] == "SHADOW_ONLY_NO_PROMOTION"


def test_counterfactual_hook_is_removed_even_when_model_forward_fails():
    class FailingBlock(ToyBlock):
        def forward(self, hidden):
            result = super().forward(hidden)
            raise RuntimeError("intentional")

    model = ToyLM()
    model.model.layers[1] = FailingBlock(16)
    adapter = LatentResidualAdapter(16, seed=3)
    inputs = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
    probe = CounterfactualSurgeryProbe(
        model,
        adapter,
        base_model_fingerprint="base",
        latent_digest="latent",
    )
    with pytest.raises(RuntimeError, match="intentional"):
        probe.run(inputs, [0.1] * 32, layer_indices=[1], gate=0.02)
    assert all(len(layer._forward_hooks) == 0 for layer in resolve_transformer_layers(model))
