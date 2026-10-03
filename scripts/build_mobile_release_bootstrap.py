from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from nolane_personal.mobile_persistent_state import (
    build_persistent_mobile_state,
    write_persistent_mobile_state,
)
from nolane_personal.product_profile import ProductProfile
from nolane_personal.state import LivingState


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
        raise RuntimeError(
            "mobile bootstrap generation requires torch"
        ) from exc
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    config = dict(payload.get("cortex_config", {}))
    latent_dim = int(config.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError(
            "factorized checkpoint is missing cortex latent_dim"
        )
    return latent_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "factorized-nolane.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    expected = str(args.checkpoint_sha256).strip().lower()
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError(
            "mobile bootstrap checkpoint SHA-256 mismatch: "
            f"expected={expected} actual={actual}"
        )

    latent_dim = latent_dim_from_checkpoint(checkpoint)
    profile = ProductProfile()
    state = LivingState(
        identity_id="bootstrap-replaced-on-first-localmobile-launch"
    )
    payload = build_persistent_mobile_state(
        source_checkpoint_sha256=actual,
        latent=[0.0] * latent_dim,
        profile=profile,
        state=state,
        memories=[],
    )
    output = write_persistent_mobile_state(
        args.output,
        payload,
        expected_source_checkpoint_sha256=actual,
        expected_latent_dim=latent_dim,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
