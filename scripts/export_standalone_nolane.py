from __future__ import annotations

import argparse
import json
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.qwen import QwenCortex
from nolane_personal.standalone_artifact import export_standalone_from_l15
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--l15", default="runtime-data/l15-native-boundary/native-nolane-boundary.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--output-dir", default="runtime-data/l16-standalone")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dataset-fingerprint", default=None)
    args = parser.parse_args()

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    qwen = QwenCortex(args.model, device=args.device)
    manifest = export_standalone_from_l15(
        qwen.model,
        args.l15,
        args.output_dir,
        latent=latent.values,
        expected_base_model_fingerprint=fingerprint,
        expected_dataset_fingerprint=args.dataset_fingerprint,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
