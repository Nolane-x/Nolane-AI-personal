from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.anchor_artifact import build_anchor_model
from nolane_personal.latent import LatentStore
from nolane_personal.native_artifact import build_native_boundary_model
from nolane_personal.native_court import run_with_decoder_call_count
from nolane_personal.native_evaluation import NativeResourceEvidence, decide_native_resources
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
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
    return tokenizer(rendered, return_tensors="pt")


def timed(torch, fn, device):
    sync(torch, device)
    start = time.perf_counter()
    result = fn()
    sync(torch, device)
    return result, (time.perf_counter()-start)*1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--native-boundary", default="runtime-data/l15-native-boundary/native-nolane-boundary.pt")
    parser.add_argument("--l14-anchor", default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--prompts", default=str(ROOT / "research/personalization-general-anchor.jsonl"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")
    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    native, native_meta = build_native_boundary_model(
        qwen.model,
        args.native_boundary,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    l14, _ = build_anchor_model(
        qwen.model,
        args.l14_anchor,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    examples = load_jsonl(args.prompts)
    samples = [
        {k: v.to(qwen.device) for k, v in dict(encode_prompt(qwen.tokenizer, row.prompt)).items()}
        for row in examples
    ]
    if not samples:
        raise SystemExit("resource prompt set is empty")

    base_times, l14_times, native_times, generation_tps = [], [], [], []
    decoder_calls = 0
    with torch.inference_mode():
        for sample in samples:
            ids = sample["input_ids"]
            _, base_ms = timed(
                torch,
                lambda: qwen.model(**sample, use_cache=False),
                qwen.device,
            )
            _, l14_ms = timed(
                torch,
                lambda: l14.forward(**sample, use_cache=False),
                qwen.device,
            )
            (_native_out, calls, _), native_ms = timed(
                torch,
                lambda: run_with_decoder_call_count(
                    qwen.model,
                    lambda: native.forward(input_ids=ids, state=None),
                ),
                qwen.device,
            )
            decoder_calls += calls

            start = time.perf_counter()
            generated, gen_calls, _ = run_with_decoder_call_count(
                qwen.model,
                lambda: native.generate(
                    input_ids=ids,
                    max_new_tokens=args.generation_tokens,
                    do_sample=False,
                    eos_token_id=None,
                ),
            )
            sync(torch, qwen.device)
            elapsed = max(time.perf_counter()-start, 1e-9)
            produced = max(1, generated.shape[1] - ids.shape[1])
            generation_tps.append(produced / elapsed)
            decoder_calls += gen_calls
            base_times.append(base_ms)
            l14_times.append(l14_ms)
            native_times.append(native_ms)

    base = statistics.median(base_times)
    l14_ms = statistics.median(l14_times)
    native_ms = statistics.median(native_times)
    evidence = NativeResourceEvidence(
        prompts=len(samples),
        baseline_median_ms=base,
        l14_median_ms=l14_ms,
        native_median_ms=native_ms,
        latency_ratio_vs_base=native_ms/max(base, 1e-9),
        latency_ratio_vs_l14=native_ms/max(l14_ms, 1e-9),
        native_tokens_per_second=statistics.median(generation_tps),
        artifact_bytes=Path(args.native_boundary).stat().st_size,
        cortex_parameters=native.trainable_parameter_count(),
        qwen_decoder_calls=decoder_calls,
    )
    decision = decide_native_resources(evidence)
    result = {
        "schema": "NOLANE-L15-NATIVE-BOUNDARY-RESOURCE-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "native_checkpoint_sha256": native_meta["checkpoint_sha256"],
        "spec_sha256": native_meta["spec_sha256"],
        "decision": decision,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite native resource evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision["status"] == "NATIVE_BOUNDARY_RESOURCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
