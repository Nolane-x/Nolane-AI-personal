from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.island_artifact import build_island_cortex
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.state_space_artifact import build_state_space_model
from nolane_personal.state_space_evaluation import StateSpaceResourceEvidence, decide_state_space_resources
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sync(torch, device) -> None:
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


def timed_forward(torch, fn, device):
    sync(torch, device)
    start = time.perf_counter()
    fn()
    sync(torch, device)
    return (time.perf_counter()-start)*1000.0


def generation_tps(torch, model, sample, *, cached: bool, max_new_tokens: int, pad_token_id: int) -> float:
    model.reset_state()
    sync(torch, next(model.model.parameters()).device)
    start = time.perf_counter()
    if cached:
        output = model.generate_cached(
            **sample,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    else:
        output = model.generate_replay_safe(
            **sample,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    sync(torch, next(model.model.parameters()).device)
    elapsed = max(time.perf_counter()-start, 1e-9)
    produced = max(1, output.shape[1] - sample["input_ids"].shape[1])
    return produced / elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--state-space", default="runtime-data/l12-state-space-cortex/selective-state-space-cortex.pt")
    parser.add_argument("--l11-islands", default="runtime-data/l11-recurrent-islands/recurrent-transformer-islands.pt")
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
    state_space, state_meta = build_state_space_model(
        qwen.model,
        args.state_space,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    l11, _l11_meta = build_island_cortex(
        qwen.model,
        args.l11_islands,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )

    examples = load_jsonl(args.prompts)
    encoded = [
        {key: value.to(qwen.device) for key, value in dict(encode_prompt(qwen.tokenizer, row.prompt)).items()}
        for row in examples
    ]
    if not encoded:
        raise SystemExit("resource prompt set is empty")

    pad = qwen.tokenizer.eos_token_id if qwen.tokenizer.eos_token_id is not None else 0
    qwen.model.eval()
    state_space.cortex.eval()
    l11.replacement.eval()
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            qwen.model(**encoded[0], use_cache=False)
            l11.forward(**encoded[0], use_cache=False)
            state_space.forward(**encoded[0], use_cache=False)

        base_times, l11_times, ss_times = [], [], []
        cached_tps, replay_tps = [], []
        for sample in encoded:
            base_times.append(timed_forward(
                torch,
                lambda: qwen.model(**sample, use_cache=False),
                qwen.device,
            ))
            l11_times.append(timed_forward(
                torch,
                lambda: l11.forward(**sample, use_cache=False),
                qwen.device,
            ))
            ss_times.append(timed_forward(
                torch,
                lambda: state_space.forward(**sample, use_cache=False),
                qwen.device,
            ))
            cached_tps.append(generation_tps(
                torch, state_space, sample,
                cached=True,
                max_new_tokens=args.generation_tokens,
                pad_token_id=pad,
            ))
            replay_tps.append(generation_tps(
                torch, state_space, sample,
                cached=False,
                max_new_tokens=args.generation_tokens,
                pad_token_id=pad,
            ))

    base = statistics.median(base_times)
    l11_ms = statistics.median(l11_times)
    ss_ms = statistics.median(ss_times)
    cached = statistics.median(cached_tps)
    replay = statistics.median(replay_tps)
    evidence = StateSpaceResourceEvidence(
        prompts=len(encoded),
        total_layers=int(qwen.model.config.num_hidden_layers),
        replaced_layers=state_space.replaced_layer_count(),
        baseline_median_ms=base,
        l11_median_ms=l11_ms,
        state_space_median_ms=ss_ms,
        latency_ratio_vs_base=ss_ms/max(base, 1e-9),
        latency_ratio_vs_l11=ss_ms/max(l11_ms, 1e-9),
        cached_tokens_per_second=cached,
        replay_safe_tokens_per_second=replay,
        cached_speedup_vs_replay=cached/max(replay, 1e-9),
        artifact_bytes=Path(args.state_space).stat().st_size,
        cortex_parameters=state_space.trainable_parameter_count(),
    )
    decision = decide_state_space_resources(evidence)
    result = {
        "schema": "NOLANE-L12-STATE-SPACE-RESOURCE-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "state_space_checkpoint_sha256": state_meta["checkpoint_sha256"],
        "plan_sha256": state_meta["plan_sha256"],
        "decision": decision,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite state-space resource evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision["status"] == "STATE_SPACE_CORTEX_RESOURCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
