from __future__ import annotations

import argparse
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.standalone_artifact import load_standalone_model


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runtime-data/l16-standalone/standalone-nolane.pt")
    parser.add_argument("--latent", default="runtime-data/living-core-shadow/latent.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-ids", required=True, help="comma-separated token ids")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    latent = LatentStore(args.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {args.latent}")

    model, metadata = load_standalone_model(
        args.checkpoint,
        latent.values,
        device=args.device,
    )
    torch = model.cortex.torch
    ids = [int(item.strip()) for item in args.input_ids.split(",") if item.strip()]
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )
    print(f"[L16 STANDALONE | {metadata['checkpoint_sha256'][:12]}]")
    print(",".join(str(int(x)) for x in output[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
