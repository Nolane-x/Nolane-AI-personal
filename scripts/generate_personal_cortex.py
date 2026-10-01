from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_artifact import build_personal_cortex
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--adapter", default="runtime-data/l6-personal-cortex/personal-cortex-adapter.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    lock = load_model_lock(args.model_lock)
    fingerprint = model_lock_fingerprint(lock)
    qwen = QwenCortex(args.model, device=args.device)
    personal, metadata = build_personal_cortex(
        qwen.model,
        args.adapter,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    try:
        rendered = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = qwen.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = qwen.tokenizer(rendered, return_tensors="pt").to(qwen.device)
    output = personal.generate(
        **dict(inputs),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=qwen.tokenizer.eos_token_id,
    )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    text = qwen.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"[UNPROMOTED PERSONAL CORTEX | {metadata['checkpoint_sha256'][:12]}]")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
