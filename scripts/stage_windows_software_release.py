from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SCHEMA = "NOLANE-V100-WINDOWS-SOFTWARE-RELEASE-V1"
AUTHORITY = "CI_SOFTWARE_RELEASE_PINNED_UPSTREAM_RUNTIME"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        file_hash = sha256_file(path).encode("ascii")
        digest.update(file_hash)
    return digest.hexdigest()


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument("--resources", default="apps/product-client/src-tauri/resources")
    parser.add_argument("--product-version", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-source-commit", required=True)
    parser.add_argument("--llama-tag", required=True)
    args = parser.parse_args()

    runtime_source = Path(args.runtime_dir).resolve()
    model_source = Path(args.model).resolve()
    llama_source = Path(args.llama_dir).resolve()
    resources = Path(args.resources).resolve()

    runtime_exe = runtime_source / "nolane-product-runtime.exe"
    llama_server = llama_source / "llama-server.exe"
    if not runtime_exe.is_file():
        raise SystemExit(f"software runtime executable missing: {runtime_exe}")
    if not model_source.is_file():
        raise SystemExit(f"software GGUF model missing: {model_source}")
    if not llama_server.is_file():
        raise SystemExit(f"llama-server executable missing: {llama_server}")

    runtime_dest = resources / "runtime"
    model_dest = resources / "model"
    clear_directory(runtime_dest)
    clear_directory(model_dest)

    shutil.copytree(runtime_source, runtime_dest, dirs_exist_ok=True)
    llama_dest = runtime_dest / "llama"
    shutil.copytree(llama_source, llama_dest, dirs_exist_ok=True)

    model_dest_path = model_dest / "Qwen_Qwen3.5-2B-Q8_0.gguf"
    shutil.copy2(model_source, model_dest_path)

    staged_runtime = runtime_dest / "nolane-product-runtime.exe"
    staged_llama = llama_dest / "llama-server.exe"

    manifest = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "product_version": str(args.product_version),
        "runtime_channel": "software-v1-gguf",
        "model_repo": str(args.model_repo),
        "model_revision": str(args.model_revision),
        "model_source_file_commit": str(args.model_source_commit),
        "model_filename": model_dest_path.name,
        "model_sha256": sha256_file(model_dest_path),
        "llama_cpp_repo": "ggml-org/llama.cpp",
        "llama_cpp_tag": str(args.llama_tag),
        "llama_server_sha256": sha256_file(staged_llama),
        "llama_runtime_tree_sha256": tree_sha256(llama_dest),
        "runtime_executable_sha256": sha256_file(staged_runtime),
        "windows_one_click_prerequisites_bundled": True,
        "release_claims": {
            "same_sha_ci_software_release": True,
            "l36_certified": False,
            "physical_device_certified": False,
        },
    }
    rendered = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    (model_dest / "software-release.json").write_text(
        rendered,
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
