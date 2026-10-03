from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.mobile_prompt_contract import (
    freeze_product_prompt_contract,
    write_product_prompt_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "freezing the mobile prompt contract requires transformers"
        ) from exc

    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_json.is_file() or not tokenizer_config.is_file():
        raise FileNotFoundError(
            "tokenizer.json and tokenizer_config.json are required"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    contract = freeze_product_prompt_contract(
        tokenizer,
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=tokenizer_config,
    )
    output = write_product_prompt_contract(
        contract,
        Path(args.output),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
