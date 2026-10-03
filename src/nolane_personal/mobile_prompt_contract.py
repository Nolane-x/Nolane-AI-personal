from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .store import canonical_json, payload_digest


SCHEMA = "NOLANE-V052-FROZEN-PRODUCT-PROMPT-CONTRACT-V1"
SYSTEM_SENTINEL = "__NOLANE_V052_SYSTEM_8E3D541F__"
USER_SENTINEL = "__NOLANE_V052_USER_7A26C19B__"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_chat_template(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template rendered an empty/non-string prompt")
    return rendered


def freeze_product_prompt_contract(
    tokenizer,
    *,
    tokenizer_json: str | Path,
    tokenizer_config_json: str | Path,
) -> dict[str, Any]:
    tokenizer_json = Path(tokenizer_json)
    tokenizer_config_json = Path(tokenizer_config_json)
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    if not tokenizer_config_json.is_file():
        raise FileNotFoundError(tokenizer_config_json)

    rendered = _render_chat_template(
        tokenizer,
        SYSTEM_SENTINEL,
        USER_SENTINEL,
    )
    if rendered.count(SYSTEM_SENTINEL) != 1:
        raise ValueError("system sentinel must occur exactly once")
    if rendered.count(USER_SENTINEL) != 1:
        raise ValueError("user sentinel must occur exactly once")

    system_index = rendered.index(SYSTEM_SENTINEL)
    user_index = rendered.index(USER_SENTINEL)
    if user_index <= system_index:
        raise ValueError("user sentinel must follow system sentinel")

    prefix = rendered[:system_index]
    after_system = rendered[
        system_index + len(SYSTEM_SENTINEL):
    ]
    relative_user = after_system.index(USER_SENTINEL)
    between = after_system[:relative_user]
    suffix = after_system[
        relative_user + len(USER_SENTINEL):
    ]

    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": "PROMPT_RENDER_CONTRACT_ONLY_NO_MODEL_AUTHORITY",
        "tokenizer_json_sha256": sha256_file(tokenizer_json),
        "tokenizer_config_json_sha256": sha256_file(tokenizer_config_json),
        "roles": ["system", "user"],
        "add_generation_prompt": True,
        "enable_thinking": False,
        "segments": {
            "prefix": prefix,
            "between": between,
            "suffix": suffix,
        },
        "probe": {
            "system_sentinel": SYSTEM_SENTINEL,
            "user_sentinel": USER_SENTINEL,
            "rendered_sha256": hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest(),
        },
    }
    contract["contract_sha256"] = payload_digest(contract)
    return contract


def verify_product_prompt_contract(
    contract: dict[str, Any],
    *,
    tokenizer_json: str | Path | None = None,
    tokenizer_config_json: str | Path | None = None,
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported product prompt contract schema")
    if contract.get("authority") != (
        "PROMPT_RENDER_CONTRACT_ONLY_NO_MODEL_AUTHORITY"
    ):
        raise ValueError("product prompt contract authority mismatch")
    supplied = str(contract.get("contract_sha256", ""))
    body = dict(contract)
    body.pop("contract_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("product prompt contract digest mismatch")
    if contract.get("roles") != ["system", "user"]:
        raise ValueError("product prompt contract role sequence mismatch")
    if contract.get("add_generation_prompt") is not True:
        raise ValueError("product prompt contract generation flag mismatch")
    if contract.get("enable_thinking") is not False:
        raise ValueError("product prompt contract thinking flag mismatch")

    segments = contract.get("segments")
    if not isinstance(segments, dict):
        raise ValueError("product prompt contract segments missing")
    for name in ("prefix", "between", "suffix"):
        if not isinstance(segments.get(name), str):
            raise ValueError(f"product prompt segment {name} must be text")

    if tokenizer_json is not None:
        if sha256_file(tokenizer_json) != contract["tokenizer_json_sha256"]:
            raise ValueError("prompt contract tokenizer.json SHA-256 mismatch")
    if tokenizer_config_json is not None:
        if (
            sha256_file(tokenizer_config_json)
            != contract["tokenizer_config_json_sha256"]
        ):
            raise ValueError(
                "prompt contract tokenizer_config.json SHA-256 mismatch"
            )

    rendered_probe = render_product_prompt(
        contract,
        SYSTEM_SENTINEL,
        USER_SENTINEL,
    )
    if (
        hashlib.sha256(rendered_probe.encode("utf-8")).hexdigest()
        != contract["probe"]["rendered_sha256"]
    ):
        raise ValueError("product prompt contract probe mismatch")
    return contract


def render_product_prompt(
    contract: dict[str, Any],
    system_text: str,
    user_text: str,
) -> str:
    segments = contract["segments"]
    return (
        segments["prefix"]
        + str(system_text)
        + segments["between"]
        + str(user_text)
        + segments["suffix"]
    )


def write_product_prompt_contract(
    contract: dict[str, Any],
    output: str | Path,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        canonical_json(contract) + "\n",
        encoding="utf-8",
    )
    return output
