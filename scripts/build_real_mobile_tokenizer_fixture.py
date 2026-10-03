from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from nolane_personal.cortex import CortexRequest
from nolane_personal.memory import MemoryRecord
from nolane_personal.mobile_prompt_contract import (
    freeze_product_prompt_contract,
    render_product_prompt,
    sha256_file,
    write_product_prompt_contract,
)
from nolane_personal.product_profile import ProductProfile
from nolane_personal.product_prompt_payload import (
    product_payload_input,
    product_user_payload,
)
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.state import (
    AffectState,
    LivingState,
    OpenThread,
    RelationshipState,
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
        repo_id,
        revision=revision,
        trust_remote_code=False,
        use_fast=True,
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

    product_cases = [
        {
            "name": "vi-direct-reply",
            "profile": ProductProfile(
                preferred_name="Tài",
                language="vi",
                response_length="compact",
                conversation_style="direct",
                initiative="active",
                memory_enabled=True,
                personal_instruction="Trả lời rõ, ngắn; đừng nói sáo rỗng.",
            ),
            "request": CortexRequest(
                mode="reply",
                intent="respond_to_user",
                user_text='Tiếp tục dự án "Nolane" nhé — giữ đúng ngữ cảnh 🚀',
                state=LivingState(
                    identity_id="identity-v053-vi",
                    affect=AffectState(
                        valence=-0.125,
                        energy=0.734,
                        playfulness=0.406,
                        concern=0.287,
                        irritation=0.019,
                    ),
                    relationship=RelationshipState(
                        closeness=0.612,
                        trust=0.845,
                        familiarity=0.553,
                        interaction_count=42,
                    ),
                    open_threads=[
                        OpenThread(
                            thread_id="vi-1",
                            topic='Android "local" / Rust',
                        ),
                        OpenThread(
                            thread_id="vi-2",
                            topic="Học toán — lượng giác",
                        ),
                        OpenThread(
                            thread_id="vi-3",
                            topic="Dự án Hira\\benchmark",
                        ),
                        OpenThread(
                            thread_id="vi-4",
                            topic="Tin AI 🚀",
                        ),
                        OpenThread(
                            thread_id="vi-5",
                            topic="fifth thread must be clipped",
                        ),
                    ],
                ),
                memories=[
                    MemoryRecord(text='memory "quoted"'),
                    MemoryRecord(text="đã chọn giao diện tối giản"),
                    MemoryRecord(text="Rust native phải khớp desktop"),
                    MemoryRecord(text="memory line\nwith newline"),
                    MemoryRecord(text="emoji 🚀"),
                    MemoryRecord(text="memory-6"),
                    MemoryRecord(text="memory-7"),
                    MemoryRecord(text="memory-8"),
                    MemoryRecord(text="memory-9 must be clipped"),
                ],
            ),
        },
        {
            "name": "en-playful-reply",
            "profile": ProductProfile(
                preferred_name="Alex",
                language="en",
                response_length="expansive",
                conversation_style="playful",
                initiative="gentle",
                memory_enabled=True,
                personal_instruction="Use concrete examples when useful.",
            ),
            "request": CortexRequest(
                mode="reply",
                intent="respond_to_user",
                user_text="Can we keep the answer technical but readable?",
                state=LivingState(
                    identity_id="identity-v053-en",
                    affect=AffectState(
                        valence=0.333,
                        energy=0.501,
                        playfulness=0.777,
                        concern=0.111,
                        irritation=0.0,
                    ),
                    relationship=RelationshipState(
                        closeness=0.444,
                        trust=0.666,
                        familiarity=0.888,
                        interaction_count=7,
                    ),
                    open_threads=[
                        OpenThread(
                            thread_id="en-1",
                            topic="mobile inference",
                        ),
                    ],
                ),
                memories=[
                    MemoryRecord(text="User prefers technical detail."),
                    MemoryRecord(text="Keep permanent UI small."),
                ],
            ),
        },
        {
            "name": "auto-natural-initiative",
            "profile": ProductProfile(
                preferred_name="",
                language="auto",
                response_length="balanced",
                conversation_style="natural",
                initiative="active",
                memory_enabled=True,
                personal_instruction="",
            ),
            "request": CortexRequest(
                mode="initiative",
                intent="follow_up:unfinished_work",
                user_text=None,
                state=LivingState(
                    identity_id="identity-v053-init",
                    affect=AffectState(
                        valence=0.0,
                        energy=0.65,
                        playfulness=0.45,
                        concern=0.0,
                        irritation=0.0,
                    ),
                    relationship=RelationshipState(
                        closeness=0.05,
                        trust=0.05,
                        familiarity=0.0,
                        interaction_count=0,
                    ),
                    open_threads=[
                        OpenThread(
                            thread_id="init-1",
                            topic="unfinished mobile runtime",
                        ),
                    ],
                ),
                memories=[],
            ),
        },
    ]
    product_rows = []
    length_tokens = {
        "compact": 96,
        "balanced": 160,
        "expansive": 256,
    }
    for case in product_cases:
        profile = case["profile"]
        request = case["request"]
        structured = product_payload_input(profile, request)
        user_payload = product_user_payload(profile, request)
        rendered = render_desktop_prompt(
            desktop,
            SYSTEM_PROMPT,
            user_payload,
        )
        frozen = render_product_prompt(
            contract,
            SYSTEM_PROMPT,
            user_payload,
        )
        if rendered != frozen:
            raise RuntimeError(
                "product payload prompt drifted from frozen prompt contract"
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
                "product payload tokenizer boundary drift"
            )
        product_rows.append(
            {
                "name": case["name"],
                "payload": structured.to_dict(),
                "system_prompt": SYSTEM_PROMPT,
                "user_payload": user_payload,
                "rendered": rendered,
                "ids": list(desktop_ids),
                "max_new_tokens": length_tokens[
                    profile.response_length
                ],
            }
        )

    product_payload_fixture = {
        "schema": "NOLANE-V053-REAL-QWEN-PRODUCT-PAYLOAD-V1",
        "repo_id": repo_id,
        "revision": revision,
        "prompt_contract_file_sha256": sha256_file(contract_path),
        "rows": product_rows,
    }
    (output / "product-payload-fixture.json").write_text(
        json.dumps(
            product_payload_fixture,
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
