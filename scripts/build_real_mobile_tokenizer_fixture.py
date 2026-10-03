from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]


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

    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            revision=revision,
        )
    )
    tokenizer_path = output / "tokenizer.json"
    tokenizer_path.write_bytes(downloaded.read_bytes())

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
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
        "vocab_size_with_added": tokenizer.get_vocab_size(with_added_tokens=True),
        "rows": rows,
    }
    (output / "fixture.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"repo_id": repo_id, "revision": revision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
