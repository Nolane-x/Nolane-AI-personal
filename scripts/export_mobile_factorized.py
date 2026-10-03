from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.mobile_factorized import export_mobile_factorized_package


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latent_dim_from_checkpoint(checkpoint: Path) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("mobile export requires torch") from exc
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    cortex = dict(payload.get("cortex_config", {}))
    latent_dim = int(cortex.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError("factorized checkpoint is missing latent_dim")
    return latent_dim


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Python-free Nolane one-token mobile package."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "factorized-nolane.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    expected = str(args.checkpoint_sha256).strip().lower()
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError(
            "factorized checkpoint SHA-256 mismatch: "
            f"expected={expected} actual={actual}"
        )

    latent_dim = latent_dim_from_checkpoint(checkpoint)
    model, meta = load_factorized_model(
        checkpoint,
        [0.0] * latent_dim,
        device="cpu",
    )
    if str(meta["checkpoint_sha256"]) != actual:
        raise RuntimeError("loaded factorized checkpoint digest drift")

    manifest = export_mobile_factorized_package(
        model,
        Path(args.output_dir).resolve(),
        source_checkpoint_sha256=actual,
    )
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
