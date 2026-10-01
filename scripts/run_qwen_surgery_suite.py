from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.surgery import CounterfactualSurgeryProbe
from nolane_personal.surgery_candidate import load_candidate, load_model_lock, model_lock_fingerprint
from nolane_personal.surgery_court import evaluate_shadow_admission


ROOT = Path(__file__).resolve().parents[1]


def _verify_model_marker(model_dir: Path, expected_revision: str) -> None:
    marker = model_dir / ".nolane-model-revision"
    if not marker.exists():
        raise SystemExit(f"missing model revision marker: {marker}; run scripts/download_model.py")
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != expected_revision:
        raise SystemExit(f"model revision mismatch: expected={expected_revision} actual={actual}")


def _parse_layers(raw: str | None):
    if raw is None:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_prompts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("prompts", [])
    if not isinstance(payload, list):
        raise SystemExit("prompt suite must be a JSON list or {'prompts': [...]}")
    prompts = [str(item).strip() for item in payload if str(item).strip()]
    if not prompts:
        raise SystemExit("prompt suite is empty")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(ROOT / "research/L5-SHADOW-PROMPTS.json"))
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--adapter", default="runtime-data/l5-adapter-candidate/latent-adapter.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--layers", default=None)
    parser.add_argument("--gate", type=float, default=0.02)
    parser.add_argument("--token-scope", choices=["last", "all"], default="last")
    parser.add_argument("--output", default="runtime-data/l5-surgery-suite.json")
    args = parser.parse_args()

    lock = load_model_lock(args.model_lock)
    expected_revision = str(lock["upstream"]["revision"])
    model_dir = Path(args.model)
    _verify_model_marker(model_dir, expected_revision)

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")
    if latent.latent_dim != 32:
        raise SystemExit(f"expected 32D latent, got {latent.latent_dim}")

    cortex = QwenCortex(model_dir, device=args.device)
    hidden_size = int(cortex.model.config.hidden_size)
    fingerprint = model_lock_fingerprint(lock)
    adapter, candidate = load_candidate(
        args.adapter,
        expected_model_lock_fingerprint=fingerprint,
        expected_hidden_size=hidden_size,
    )
    adapter.to(cortex.device).eval()
    probe = CounterfactualSurgeryProbe(
        cortex.model,
        adapter,
        base_model_fingerprint=fingerprint,
        latent_digest=latent.digest,
    )

    receipts = []
    for prompt in _load_prompts(Path(args.prompts)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = cortex.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = cortex.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        model_inputs = cortex.tokenizer(rendered, return_tensors="pt").to(cortex.device)
        _served_baseline, receipt = probe.run(
            dict(model_inputs),
            latent.values,
            layer_indices=_parse_layers(args.layers),
            gate=args.gate,
            token_scope=args.token_scope,
        )
        receipts.append(receipt)

    admission = evaluate_shadow_admission(receipts)
    result = {
        "schema": "NOLANE-L5-QWEN-SUITE-V1",
        "authority": "SHADOW_ONLY_NO_PROMOTION",
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "prompt_count": len(receipts),
        "receipts": [receipt.to_dict() for receipt in receipts],
        "admission": asdict(admission),
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite suite result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["admission"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if admission.status == "SHADOW_ADMISSION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
