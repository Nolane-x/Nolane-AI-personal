from __future__ import annotations

import argparse
import json

from nolane_personal.mobile_release import (
    stage_localmobile_release_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage an L36-authorized Python-free LocalMobile Android bundle."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--ceremony", required=True)
    parser.add_argument(
        "--resources",
        default="apps/product-client/src-tauri/resources/mobile",
    )
    args = parser.parse_args()

    manifest = stage_localmobile_release_bundle(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        tokenizer_dir=args.tokenizer_dir,
        ceremony_path=args.ceremony,
        resources=args.resources,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
