from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_asset_staging_requires_exact_model_and_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "nolane-product-runtime.exe").write_bytes(b"runtime")
    (runtime / "support.dll").write_bytes(b"support")

    model = tmp_path / "factorized-nolane.pt"
    model.write_bytes(b"approved-model")

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text(
        '{"chat_template":"test"}',
        encoding="utf-8",
    )
    (tokenizer / "tokenizer.json").write_text(
        '{"version":"1.0"}',
        encoding="utf-8",
    )

    resources = tmp_path / "resources"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/stage_product_release_assets.py",
            "--runtime-dir",
            str(runtime),
            "--model",
            str(model),
            "--model-sha256",
            sha(model),
            "--tokenizer-dir",
            str(tokenizer),
            "--resources",
            str(resources),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (resources / "runtime" / "nolane-product-runtime.exe").is_file()
    assert (resources / "model" / "factorized-nolane.pt").read_bytes() == b"approved-model"
    manifest = json.loads(
        (resources / "release-assets.json").read_text(encoding="utf-8")
    )
    assert manifest["model_checkpoint_sha256"] == sha(model)
    assert manifest["qwen_model_object_required"] is False
    assert manifest["windows_one_click_prerequisites_bundled"] is True


def test_release_asset_staging_fails_closed_on_model_hash_mismatch(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "nolane-product-runtime.exe").write_bytes(b"runtime")
    model = tmp_path / "factorized-nolane.pt"
    model.write_bytes(b"wrong-model")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/stage_product_release_assets.py",
            "--runtime-dir",
            str(runtime),
            "--model",
            str(model),
            "--model-sha256",
            "0" * 64,
            "--tokenizer-dir",
            str(tokenizer),
            "--resources",
            str(tmp_path / "resources"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "SHA-256 mismatch" in (completed.stdout + completed.stderr)
