from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if match is None:
        raise SystemExit("pyproject.toml project version missing")
    python_version = match.group(1)

    package = json.loads(
        (ROOT / "apps/product-client/package.json").read_text(encoding="utf-8")
    )
    tauri = json.loads(
        (ROOT / "apps/product-client/src-tauri/tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    cargo = (
        ROOT / "apps/product-client/src-tauri/Cargo.toml"
    ).read_text(encoding="utf-8")
    cargo_match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        cargo,
        re.MULTILINE,
    )
    if cargo_match is None:
        raise SystemExit("Cargo.toml package version missing")

    versions = {
        "python": python_version,
        "npm": str(package.get("version", "")),
        "cargo": cargo_match.group(1),
        "tauri": str(tauri.get("version", "")),
    }
    unique = set(versions.values())
    if "" in unique or len(unique) != 1:
        raise SystemExit(
            "product version mismatch: "
            + ", ".join(f"{name}={value}" for name, value in versions.items())
        )

    version = unique.pop()
    print(json.dumps({"status": "pass", "version": version}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
