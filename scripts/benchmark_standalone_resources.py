from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.native_artifact import build_native_boundary_model
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.standalone_artifact import load_standalone_model
from nolane_personal.standalone_evaluation import (
    StandaloneResourceEvidence,
    decide_standalone_resources,
)
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sync(torch, device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def encode_prompt(tokenizer, prompt):
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
    out = fn()
    sync(torch, device)
    return out, (time.perf_counter()-start)*1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--l15", default="runtime-data/l15-native-boundary/native-nolane-boundary.pt")
    parser.add_argument("--standalone", default="runtime-data/l16-standalone/standalone-nolane.pt")
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
    l15, _l15_meta = build_native_boundary_model(
        qwen.model,
        args.l15,
        latent.values,
        expected_base_model_fingerprint=fingerprint,
    )
    standalone, meta = load_standalone_model(
        args.standalone,
        latent.values,
        device=qwen.device,
        expected_source_base_model_fingerprint=fingerprint,
    )
    samples = [
        {
            k: v.to(qwen.device)
            for k, v in dict(encode_prompt(qwen.tokenizer, row.prompt)).items()
        }
        for row in load_jsonl(args.prompts)
    ]
    if not samples:
        raise SystemExit("resource prompt set is empty")

    l15_times, standalone_times, tps = [], [], []
    with torch.inference_mode():
        for sample in samples:
            ids = sample["input_ids"]
            _, l15_ms = timed(
                torch,
                lambda: l15.forward(input_ids=ids, state=None),
                qwen.device,
            )
            _, standalone_ms = timed(
                torch,
                lambda: standalone.forward(input_ids=ids, state=None),
                qwen.device,
            )
            start = time.perf_counter()
            generated = standalone.generate(
                input_ids=ids,
                max_new_tokens=args.generation_tokens,
                do_sample=False,
                eos_token_id=None,
            )
            sync(torch, qwen.device)
            elapsed = max(time.perf_counter()-start, 1e-9)
            produced = max(1, generated.shape[1]-ids.shape[1])
            tps.append(produced/elapsed)
            l15_times.append(l15_ms)
            standalone_times.append(standalone_ms)

    l15_median = statistics.median(l15_times)
    standalone_median = statistics.median(standalone_times)
    manifest_path = Path(args.standalone).with_name("standalone-nolane-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_payload_bytes = (
        int(manifest["source_boundary_bytes"])
        + int(manifest["cortex_parameters"]) * 4
    )
    evidence = StandaloneResourceEvidence(
        prompts=len(samples),
        l15_median_ms=l15_median,
        standalone_median_ms=standalone_median,
        latency_ratio_vs_l15=standalone_median/max(l15_median, 1e-9),
        standalone_tokens_per_second=statistics.median(tps),
        checkpoint_bytes=Path(args.standalone).stat().st_size,
        source_boundary_bytes=source_payload_bytes,
        checkpoint_to_boundary_ratio=Path(args.standalone).stat().st_size/max(
            source_payload_bytes,
            1,
        ),
        runtime_qwen_model_objects=int(hasattr(standalone, "qwen_model")),
    )
    decision = decide_standalone_resources(evidence)
    result = {
        "schema": "NOLANE-L16-STANDALONE-RESOURCE-EVAL-V1",
        "standalone_checkpoint_sha256": meta["checkpoint_sha256"],
        "source_l15_checkpoint_sha256": meta["source_l15_checkpoint_sha256"],
        "decision": decision,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite standalone resource evidence: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered+"\n", encoding="utf-8")
    print(rendered)
    return 0 if decision["status"] == "STANDALONE_RESOURCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
