from __future__ import annotations

from contextlib import nullcontext

import pytest

from nolane_personal.cortex import CortexRequest
from nolane_personal.product_cortex import FactorizedProductCortex
from nolane_personal.product_profile import ProductProfile
from nolane_personal.state import LivingState


class FakeTensor:
    def __init__(self, width: int):
        self.ndim = 2
        self.shape = (1, width)

    def to(self, _device):
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, _text, *, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": FakeTensor(3)}


class FakeModel:
    def __init__(self, generated_width: int):
        self.generated_width = generated_width

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 1
        assert kwargs["do_sample"] is False
        return FakeTensor(self.generated_width)


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def cortex_with_generated_width(width: int):
    cortex = FactorizedProductCortex.__new__(FactorizedProductCortex)
    cortex.tokenizer = FakeTokenizer()
    cortex.model = FakeModel(width)
    cortex.torch = FakeTorch()
    cortex.device = "cpu"
    cortex.checkpoint_sha256 = "a" * 64
    return cortex


def test_product_generate_forwards_exact_seeded_sampling_policy():
    torch = pytest.importorskip("torch")
    captured = {}

    class ProductTokenizer:
        eos_token_id = 2

        def apply_chat_template(
            self,
            _messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking=False,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "rendered product prompt"

        def __call__(self, text, *, return_tensors):
            assert text == "rendered product prompt"
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor(
                    [[1, 4, 6]],
                    dtype=torch.long,
                )
            }

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            assert tokens.tolist() == [7]
            return "seeded reply"

    class ProductModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor(
                [[1, 4, 6, 7]],
                dtype=torch.long,
            )

    fixed_seed = 0x123456789ABCDEF0
    cortex = FactorizedProductCortex.__new__(FactorizedProductCortex)
    cortex.tokenizer = ProductTokenizer()
    cortex.model = ProductModel()
    cortex.torch = torch
    cortex.device = "cpu"
    cortex.profile_getter = lambda: ProductProfile(
        response_length="compact",
    )
    cortex.sampling_seed_getter = lambda: fixed_seed

    reply = cortex.generate(
        CortexRequest(
            mode="reply",
            intent="conversation",
            user_text="hello",
            state=LivingState(),
        )
    )

    assert reply.utterance == "seeded reply"
    assert captured["max_new_tokens"] == 96
    assert captured["do_sample"] is True
    assert captured["temperature"] == pytest.approx(0.78)
    assert captured["top_p"] == pytest.approx(0.90)
    assert captured["sampling_seed"] == fixed_seed
    assert captured["eos_token_id"] == 2
    assert captured["pad_token_id"] == 2


def test_factorized_product_self_test_requires_real_generated_token():
    cortex = cortex_with_generated_width(4)
    result = cortex.self_test()
    assert result == {
        "status": "PASS",
        "generated_tokens": 1,
        "checkpoint_sha256": "a" * 64,
        "device": "cpu",
    }


def test_factorized_product_self_test_blocks_zero_new_tokens():
    cortex = cortex_with_generated_width(3)
    with pytest.raises(RuntimeError, match="produced no new token"):
        cortex.self_test()


def test_factorized_product_self_test_blocks_invalid_model_shape():
    cortex = cortex_with_generated_width(4)
    cortex.model.generate = lambda **_kwargs: type(
        "BadTensor",
        (),
        {"ndim": 1, "shape": (4,)},
    )()
    with pytest.raises(RuntimeError, match="invalid shape"):
        cortex.self_test()
