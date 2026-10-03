from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from nolane_personal.promotion_ceremony import verify_promotion_ceremony_receipt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        path.mkdir(parents=True)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"source directory missing: {source}")
    clear_directory(destination)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--ceremony", required=True)
    parser.add_argument("--tokenizer-archive-sha256", default="")
    parser.add_argument(
        "--resources",
        default="apps/product-client/src-tauri/resources",
    )
    args = parser.parse_args()

    runtime_source = Path(args.runtime_dir)
    model_source = Path(args.model)
    tokenizer_source = Path(args.tokenizer_dir)
    ceremony_source = Path(args.ceremony)
    resources = Path(args.resources)
    runtime_dest = resources / "runtime"
    model_dest = resources / "model"
    tokenizer_dest = resources / "tokenizer"

    runtime_exe = runtime_source / "nolane-product-runtime.exe"
    if not runtime_exe.is_file():
        raise SystemExit(
            f"PyInstaller runtime executable missing: {runtime_exe}"
        )
    if not model_source.is_file():
        raise SystemExit(f"release model missing: {model_source}")

    expected_model = args.model_sha256.strip().lower()
    actual_model = sha256_file(model_source)
    if len(expected_model) != 64 or actual_model != expected_model:
        raise SystemExit(
            "release model SHA-256 mismatch: "
            f"expected={expected_model} actual={actual_model}"
        )

    if not ceremony_source.is_file():
        raise SystemExit(f"promotion ceremony missing: {ceremony_source}")
    ceremony = json.loads(ceremony_source.read_text(encoding="utf-8"))
    verify_promotion_ceremony_receipt(ceremony, require_complete=True)
    if ceremony["candidate_checkpoint_sha256"] != actual_model:
        raise SystemExit(
            "promotion ceremony checkpoint mismatch: "
            f"ceremony={ceremony['candidate_checkpoint_sha256']} "
            f"model={actual_model}"
        )

    required_tokenizer = ("tokenizer_config.json", "tokenizer.json")
    missing = [
        name for name in required_tokenizer
        if not (tokenizer_source / name).is_file()
    ]
    if missing:
        raise SystemExit(
            "tokenizer release assets missing: " + ", ".join(missing)
        )

    copy_tree(runtime_source, runtime_dest)
    clear_directory(model_dest)
    shutil.copy2(model_source, model_dest / "factorized-nolane.pt")
    shutil.copy2(
        ceremony_source,
        model_dest / "promotion-ceremony.json",
    )
    copy_tree(tokenizer_source, tokenizer_dest)

    manifest = {
        "schema": "NOLANE-V042-WINDOWS-RELEASE-ASSETS-V1",
        "runtime_executable_sha256": sha256_file(
            runtime_dest / "nolane-product-runtime.exe"
        ),
        "model_checkpoint_sha256": actual_model,
        "promotion_ceremony_sha256": ceremony["ceremony_sha256"],
        "promotion_authorization_sha256": ceremony["authorization_sha256"],
        "tokenizer_config_sha256": sha256_file(
            tokenizer_dest / "tokenizer_config.json"
        ),
        "tokenizer_json_sha256": sha256_file(
            tokenizer_dest / "tokenizer.json"
        ),
        "tokenizer_archive_sha256": (
            args.tokenizer_archive_sha256.strip().lower() or None
        ),
        "windows_one_click_prerequisites_bundled": True,
        "model_runtime": "factorized-nolane",
        "qwen_model_object_required": False,
    }
    resources.mkdir(parents=True, exist_ok=True)
    rendered_manifest = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n"
    (resources / "release-assets.json").write_text(
        rendered_manifest,
        encoding="utf-8",
    )
    (model_dest / "release-assets.json").write_text(
        rendered_manifest,
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
