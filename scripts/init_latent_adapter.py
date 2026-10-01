from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.surgery import LatentAdapterConfig
from nolane_personal.surgery_candidate import create_candidate, load_model_lock


ROOT = Path(__file__).resolve().parents[1]


def _verify_model_marker(model_dir: Path, expected_revision: str) -> None:
    marker = model_dir / ".nolane-model-revision"
    if not marker.exists():
        raise SystemExit(f"missing model revision marker: {marker}; run scripts/download_model.py")
    actual = marker.read_text(encoding="utf-8").strip()
    if actual != expected_revision:
        raise SystemExit(f"model revision mismatch: expected={expected_revision} actual={actual}")


def main() -> int:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise SystemExit("Install Qwen support with: pip install -e '.[qwen]'") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--output-dir", default="runtime-data/l5-adapter-candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bottleneck-dim", type=int, default=16)
    parser.add_argument("--max-abs-gate", type=float, default=0.10)
    args = parser.parse_args()

    lock = load_model_lock(args.model_lock)
    expected_revision = str(lock["upstream"]["revision"])
    model_dir = Path(args.model)
    _verify_model_marker(model_dir, expected_revision)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    hidden_size = int(config.hidden_size)

    manifest = create_candidate(
        args.output_dir,
        hidden_size=hidden_size,
        model_lock=lock,
        config=LatentAdapterConfig(
            latent_dim=32,
            bottleneck_dim=args.bottleneck_dim,
            max_abs_gate=args.max_abs_gate,
        ),
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
