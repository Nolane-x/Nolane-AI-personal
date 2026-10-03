from __future__ import annotations

import json

import pytest

from nolane_personal.mobile_prompt_contract import (
    SYSTEM_SENTINEL,
    USER_SENTINEL,
    freeze_product_prompt_contract,
    render_product_prompt,
    verify_product_prompt_contract,
)


class FakeTokenizer:
    def __init__(self, mode: str = "normal"):
        self.mode = mode

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        system = messages[0]["content"]
        user = messages[1]["content"]
        if self.mode == "duplicate-system":
            system = system + system
        if self.mode == "reverse":
            return f"<u>{user}</u><s>{system}</s><a>"
        return f"<s>{system}</s><u>{user}</u><a>"


def tokenizer_files(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    config = tmp_path / "tokenizer_config.json"
    tokenizer.write_text('{"model":{"type":"fixture"}}\n', encoding="utf-8")
    config.write_text('{"chat_template":"fixture"}\n', encoding="utf-8")
    return tokenizer, config


def test_frozen_prompt_contract_roundtrip_and_dynamic_render(tmp_path):
    tokenizer_json, config_json = tokenizer_files(tmp_path)
    contract = freeze_product_prompt_contract(
        FakeTokenizer(),
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=config_json,
    )

    verified = verify_product_prompt_contract(
        contract,
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=config_json,
    )
    assert verified == contract
    assert contract["segments"] == {
        "prefix": "<s>",
        "between": "</s><u>",
        "suffix": "</u><a>",
    }

    rendered = render_product_prompt(
        contract,
        "system text — Việt",
        "user <content> 🚀",
    )
    assert rendered == (
        "<s>system text — Việt</s>"
        "<u>user <content> 🚀</u><a>"
    )


def test_prompt_contract_binds_tokenizer_files(tmp_path):
    tokenizer_json, config_json = tokenizer_files(tmp_path)
    contract = freeze_product_prompt_contract(
        FakeTokenizer(),
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=config_json,
    )

    tokenizer_json.write_text('{"model":{"type":"changed"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer.json SHA-256 mismatch"):
        verify_product_prompt_contract(
            contract,
            tokenizer_json=tokenizer_json,
            tokenizer_config_json=config_json,
        )


def test_prompt_contract_detects_contract_tamper(tmp_path):
    tokenizer_json, config_json = tokenizer_files(tmp_path)
    contract = freeze_product_prompt_contract(
        FakeTokenizer(),
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=config_json,
    )
    tampered = json.loads(json.dumps(contract))
    tampered["segments"]["suffix"] += "tamper"
    with pytest.raises(ValueError, match="contract digest mismatch"):
        verify_product_prompt_contract(tampered)


def test_prompt_contract_requires_unique_ordered_sentinels(tmp_path):
    tokenizer_json, config_json = tokenizer_files(tmp_path)

    with pytest.raises(ValueError, match="system sentinel must occur exactly once"):
        freeze_product_prompt_contract(
            FakeTokenizer("duplicate-system"),
            tokenizer_json=tokenizer_json,
            tokenizer_config_json=config_json,
        )

    with pytest.raises(ValueError, match="user sentinel must follow system sentinel"):
        freeze_product_prompt_contract(
            FakeTokenizer("reverse"),
            tokenizer_json=tokenizer_json,
            tokenizer_config_json=config_json,
        )


def test_sentinels_are_distinct_and_not_prefixes():
    assert SYSTEM_SENTINEL != USER_SENTINEL
    assert SYSTEM_SENTINEL not in USER_SENTINEL
    assert USER_SENTINEL not in SYSTEM_SENTINEL
