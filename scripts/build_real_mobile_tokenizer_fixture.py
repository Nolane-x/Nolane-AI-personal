from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from nolane_personal.mobile_prompt_contract import (
    freeze_product_prompt_contract,
    render_product_prompt,
    sha256_file,
    write_product_prompt_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def render_desktop_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    lock = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))
    upstream = lock["upstream"]
    repo_id = str(upstream["repo_id"])
    revision = str(upstream["revision"])

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for filename in ("tokenizer.json", "tokenizer_config.json"):
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        )
        (output / filename).write_bytes(downloaded.read_bytes())

    tokenizer_path = output / "tokenizer.json"
    tokenizer_config_path = output / "tokenizer_config.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    desktop = AutoTokenizer.from_pretrained(
        str(output),
        local_files_only=True,
        trust_remote_code=False,
    )

    samples = [
        "Xin chào, hôm nay bạn thế nào?",
        "A persistent companion should remember context.",
        "Hải Phòng — 2026 🚀",
        "code: fn main() { println!(\"Nolane\"); }",
    ]
    rows = []
    for text in samples:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        rows.append(
            {
                "text": text,
                "ids": ids,
                "decoded": tokenizer.decode(ids, skip_special_tokens=True),
            }
        )

    payload = {
        "schema": "NOLANE-V051-REAL-QWEN-TOKENIZER-FIXTURE-V1",
        "repo_id": repo_id,
        "revision": revision,
        "vocab_size_with_added": tokenizer.get_vocab_size(
            with_added_tokens=True
        ),
        "rows": rows,
    }
    (output / "fixture.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contract = freeze_product_prompt_contract(
        desktop,
        tokenizer_json=tokenizer_path,
        tokenizer_config_json=tokenizer_config_path,
    )
    contract_path = write_product_prompt_contract(
        contract,
        output / "prompt-contract.json",
    )

    prompt_pairs = [
        (
            "You are Nolane. Reply naturally and concisely.",
            "Xin chào, hôm nay cậu thấy thế nào?",
        ),
        (
            "System line 1\nSystem line 2 — persistent.",
            "English + Việt + emoji 🚀\nSecond line.",
        ),
        (
            "Do not invent memories.",
            "code: if (ready) { return \"ok\"; }",
        ),
    ]
    prompt_rows = []
    for system_text, user_text in prompt_pairs:
        rendered = render_desktop_prompt(
            desktop,
            system_text,
            user_text,
        )
        frozen = render_product_prompt(
            contract,
            system_text,
            user_text,
        )
        if frozen != rendered:
            raise RuntimeError(
                "frozen prompt contract drifted from desktop chat template"
            )

        desktop_ids = desktop(
            rendered,
            add_special_tokens=True,
        )["input_ids"]
        rust_boundary_ids = tokenizer.encode(
            rendered,
            add_special_tokens=False,
        ).ids
        if list(desktop_ids) != list(rust_boundary_ids):
            raise RuntimeError(
                "desktop tokenizer special-token behavior does not match "
                "native tokenizer boundary"
            )
        prompt_rows.append(
            {
                "system": system_text,
                "user": user_text,
                "rendered": rendered,
                "ids": list(desktop_ids),
            }
        )

    prompt_payload = {
        "schema": "NOLANE-V052-REAL-QWEN-PROMPT-FIXTURE-V1",
        "repo_id": repo_id,
        "revision": revision,
        "prompt_contract_file_sha256": sha256_file(contract_path),
        "rows": prompt_rows,
    }
    (output / "prompt-fixture.json").write_text(
        json.dumps(
            prompt_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": revision,
                "prompt_contract_file_sha256": sha256_file(contract_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
