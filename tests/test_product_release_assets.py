from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from nolane_personal.promotion_ceremony import (
    AUTHORITY as CEREMONY_AUTHORITY,
    SCHEMA as CEREMONY_SCHEMA,
)
from nolane_personal.store import payload_digest


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()




def write_complete_ceremony(path: Path, checkpoint_sha256: str) -> dict:
    t0 = "2026-10-02T12:00:00+00:00"
    t1 = "2026-10-02T12:01:00+00:00"
    t2 = "2026-10-02T12:02:00+00:00"
    t3 = "2026-10-02T12:03:00+00:00"
    body = {
        "schema": CEREMONY_SCHEMA,
        "authority": CEREMONY_AUTHORITY,
        "status": "COMPLETE",
        "reasons": [],
        "authorization_sha256": "1" * 64,
        "multicycle_chain_sha256": "2" * 64,
        "long_horizon_retention_court_sha256": "3" * 64,
        "candidate_checkpoint_sha256": checkpoint_sha256,
        "pointer_sha256": "4" * 64,
        "serving_convergence_sha256": "5" * 64,
        "transaction_id": "tx-release-test",
        "pointer_generation": 1,
        "authorization_issued_at": t0,
        "authorization_expires_at": "2026-10-02T13:00:00+00:00",
        "transaction_prepared_at": t0,
        "transaction_committed_at": t1,
        "pointer_created_at": t1,
        "serving_convergence_assessed_at": t2,
        "ceremony_at": t3,
    }
    body["ceremony_sha256"] = payload_digest(body)
    path.write_text(
        json.dumps(body, sort_keys=True),
        encoding="utf-8",
    )
    return body

def test_release_asset_staging_requires_exact_model_and_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "nolane-product-runtime.exe").write_bytes(b"runtime")
    (runtime / "support.dll").write_bytes(b"support")

    model = tmp_path / "factorized-nolane.pt"
    model.write_bytes(b"approved-model")

    ceremony = tmp_path / "ceremony.json"
    ceremony_payload = write_complete_ceremony(ceremony, sha(model))

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
            "--ceremony",
            str(ceremony),
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
    assert manifest["promotion_ceremony_sha256"] == ceremony_payload["ceremony_sha256"]
    assert manifest["qwen_model_object_required"] is False
    assert manifest["windows_one_click_prerequisites_bundled"] is True


def test_release_asset_staging_fails_closed_on_model_hash_mismatch(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "nolane-product-runtime.exe").write_bytes(b"runtime")
    model = tmp_path / "factorized-nolane.pt"
    model.write_bytes(b"wrong-model")
    ceremony = tmp_path / "ceremony.json"
    write_complete_ceremony(ceremony, sha(model))
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
            "--ceremony",
            str(ceremony),
            "--resources",
            str(tmp_path / "resources"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "SHA-256 mismatch" in (completed.stdout + completed.stderr)



def test_release_asset_staging_rejects_unpromoted_model_even_with_correct_sha(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "nolane-product-runtime.exe").write_bytes(b"runtime")
    model = tmp_path / "factorized-nolane.pt"
    model.write_bytes(b"candidate-model")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    ceremony = tmp_path / "ceremony.json"
    payload = write_complete_ceremony(ceremony, "f" * 64)
    assert payload["candidate_checkpoint_sha256"] != sha(model)

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
            "--ceremony",
            str(ceremony),
            "--resources",
            str(tmp_path / "resources"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ceremony checkpoint mismatch" in (
        completed.stdout + completed.stderr
    )
