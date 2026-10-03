from __future__ import annotations

from contextlib import nullcontext

import pytest

from nolane_personal.product_cortex import FactorizedProductCortex


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
