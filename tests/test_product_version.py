from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_product_versions(root: Path, *, python: str, tauri: str, package: str):
    (root / "apps" / "product-client" / "src-tauri").mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "apps" / "product-client").mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "' + python + '"\n',
        encoding="utf-8",
    )
    (root / "apps" / "product-client" / "src-tauri" / "tauri.conf.json").write_text(
        json.dumps({"version": tauri}),
        encoding="utf-8",
    )
    (root / "apps" / "product-client" / "package.json").write_text(
        json.dumps({"version": package}),
        encoding="utf-8",
    )


def run_version_court(root: Path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_product_version.py"),
            "--root",
            str(root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_product_version_court_passes_only_when_all_surfaces_match(tmp_path):
    write_product_versions(
        tmp_path,
        python="0.48.0",
        tauri="0.48.0",
        package="0.48.0",
    )
    completed = run_version_court(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "package": "0.48.0",
        "python": "0.48.0",
        "tauri": "0.48.0",
    }


def test_product_version_court_blocks_stale_desktop_version(tmp_path):
    write_product_versions(
        tmp_path,
        python="0.48.0",
        tauri="0.42.0",
        package="0.48.0",
    )
    completed = run_version_court(tmp_path)
    assert completed.returncode != 0
    assert "product version mismatch" in (completed.stdout + completed.stderr)
    assert "tauri=0.42.0" in (completed.stdout + completed.stderr)
