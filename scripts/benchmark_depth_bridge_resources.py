from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from nolane_personal.depth_bridge_artifact import build_bridge_cortex
from nolane_personal.depth_bridge_resources import (
    DepthBridgeResourceEvidence,
    decide_depth_bridge_resources,
)
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--depth-bridge", default="runtime-data/l8-depth-bridge/depth-bridge.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--prompts", default=str(ROOT / "research/personalization-general-anchor.jsonl"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=2)
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
    depth, metadata = build_bridge_cortex(
        qwen.model,
        args.depth_bridge,
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

    qwen.model.eval()
    depth.bridge.eval()
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            sample = encoded[0]
            qwen.model(**sample, use_cache=False)
            depth.forward(**sample, use_cache=False)

        base_times = []
        depth_times = []
        for sample in encoded:
            sync(torch, qwen.device)
            start = time.perf_counter()
            qwen.model(**sample, use_cache=False)
            sync(torch, qwen.device)
            base_times.append((time.perf_counter() - start) * 1000.0)

            sync(torch, qwen.device)
            start = time.perf_counter()
            depth.forward(**sample, use_cache=False)
            sync(torch, qwen.device)
            depth_times.append((time.perf_counter() - start) * 1000.0)

    base_median = statistics.median(base_times)
    depth_median = statistics.median(depth_times)
    artifact = Path(args.depth_bridge)
    evidence = DepthBridgeResourceEvidence(
        prompts=len(encoded),
        baseline_median_ms=base_median,
        depth_bridge_median_ms=depth_median,
        latency_ratio_vs_base=depth_median / max(base_median, 1e-9),
        artifact_bytes=artifact.stat().st_size,
        bridge_parameters=depth.trainable_parameter_count(),
    )
    decision = decide_depth_bridge_resources(evidence)
    result = {
        "schema": "NOLANE-L8-DEPTH-BRIDGE-RESOURCE-EVAL-V1",
        "authority": "EVALUATION_ONLY_UNPROMOTED",
        "bridge_checkpoint_sha256": metadata["checkpoint_sha256"],
        "decision": asdict(decision),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite resource evaluation: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if decision.status == "DEPTH_BRIDGE_RESOURCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
