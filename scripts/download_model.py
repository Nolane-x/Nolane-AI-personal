from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "model.lock.json"


def load_lock(path: Path = LOCK_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the exact Qwen scaffold pinned for Nolane AI Personal."
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=LOCK_PATH,
        help="Model lock to download (defaults to the active product scaffold).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Destination directory; defaults to the path declared by the lock.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force Hugging Face Hub to re-download files.",
    )
    args = parser.parse_args()

    lock_path = args.lock.resolve()
    lock = load_lock(lock_path)
    upstream = lock["upstream"]
    local = lock["local"]
    repo_id = upstream["repo_id"]
    revision = upstream["revision"]
    target = (
        args.target.resolve()
        if args.target is not None
        else (ROOT / local["path"]).resolve()
    )
    target.mkdir(parents=True, exist_ok=True)

    print(f"[Nolane AI Personal] lock  : {lock_path}")
    print(f"[Nolane AI Personal] model : {repo_id}")
    print(f"[Nolane AI Personal] commit: {revision}")
    print(f"[Nolane AI Personal] target: {target}")

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    if info.sha != revision:
        raise RuntimeError(
            f"Revision verification failed: requested {revision}, resolved {info.sha}"
        )

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        force_download=args.force,
    )

    marker = target / ".nolane-model-revision"
    marker.write_text(revision + "\n", encoding="utf-8")

    print(f"[Nolane AI Personal] ready: {snapshot_path}")
    print("[Nolane AI Personal] exact upstream revision verified.")


if __name__ == "__main__":
    main()
