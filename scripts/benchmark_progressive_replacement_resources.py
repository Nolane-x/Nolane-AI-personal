from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.progressive_artifact import build_progressive_cortex
from nolane_personal.progressive_evaluation import ProgressiveResourceEvidence, decide_progressive_resources
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sync(torch, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def encode_prompt(tokenizer, prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, return_tensors="pt")


def tokens_per_second(torch, cortex, sample, *, cached: bool, max_new_tokens: int) -> float:
    cortex.reset_state()
    sync(torch, next(cortex.model.parameters()).device)
    start = time.perf_counter()
    if cached:
        output = cortex.generate_cached(
            **sample,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
    else:
        output = cortex.generate_replay_safe(
            **sample,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=0,
        )
    sync(torch, next(cortex.model.parameters()).device)
    elapsed = max(time.perf_counter()-start, 1e-9)
    produced = max(1, output.shape[1] - sample["input_ids"].shape[1])
    return produced / elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--progressive", default="runtime-data/l10-progressive-replacement/progressive-block-replacement.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--prompts", default=str(ROOT / "research/personalization-general-anchor.jsonl"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install Qwen support with: pip install -e '.[qwen]'") from exc

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    cortex, meta = build_progressive_cortex(
        qwen.model,
        args.progressive,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    examples = load_jsonl(args.prompts)
    encoded = [
        {k: v.to(qwen.device) for k, v in dict(encode_prompt(qwen.tokenizer, row.prompt)).items()}
        for row in examples
    ]
    if not encoded:
        raise SystemExit("resource prompt set is empty")

    qwen.model.eval()
    cortex.replacement.eval()
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            qwen.model(**encoded[0], use_cache=False)
            cortex.forward(**encoded[0], use_cache=False)

        base_times, progressive_times = [], []
        cached_tps, replay_tps = [], []
        for sample in encoded:
            sync(torch, qwen.device)
            start = time.perf_counter()
            qwen.model(**sample, use_cache=False)
            sync(torch, qwen.device)
            base_times.append((time.perf_counter()-start)*1000.0)

            sync(torch, qwen.device)
            start = time.perf_counter()
            cortex.forward(**sample, use_cache=False)
            sync(torch, qwen.device)
            progressive_times.append((time.perf_counter()-start)*1000.0)

            cached_tps.append(tokens_per_second(
                torch, cortex, sample, cached=True, max_new_tokens=args.generation_tokens
            ))
            replay_tps.append(tokens_per_second(
                torch, cortex, sample, cached=False, max_new_tokens=args.generation_tokens
            ))

    base = statistics.median(base_times)
    candidate = statistics.median(progressive_times)
    cached = statistics.median(cached_tps)
    replay = statistics.median(replay_tps)
    evidence = ProgressiveResourceEvidence(
        prompts=len(encoded),
        total_layers=int(qwen.model.config.num_hidden_layers),
        replaced_layers=len(cortex.config.layer_indices),
        baseline_median_ms=base,
        progressive_median_ms=candidate,
        latency_ratio_vs_base=candidate/max(base, 1e-9),
        cached_tokens_per_second=cached,
        replay_safe_tokens_per_second=replay,
        cached_speedup_vs_replay=cached/max(replay, 1e-9),
        artifact_bytes=Path(args.progressive).stat().st_size,
        replacement_parameters=cortex.trainable_parameter_count(),
    )
    decision = decide_progressive_resources(evidence)
    result = {
        "schema": "NOLANE-L10-PROGRESSIVE-REPLACEMENT-RESOURCE-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "progressive_checkpoint_sha256": meta["checkpoint_sha256"],
        "plan_sha256": meta["plan_sha256"],
        "decision": decision,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite resource evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision["status"] == "PROGRESSIVE_REPLACEMENT_RESOURCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
