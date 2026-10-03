from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def toml_section_version(text: str, section: str, *, source: str) -> str:
    section_match = re.search(
        rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    if section_match is None:
        raise ValueError(f"{source}: [{section}] section not found")
    version_match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        section_match.group(1),
    )
    if version_match is None:
        raise ValueError(f"{source}: version not found in [{section}]")
    return version_match.group(1)


def product_versions(root: Path) -> dict[str, str]:
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    cargo_text = (
        root / "apps/product-client/src-tauri/Cargo.toml"
    ).read_text(encoding="utf-8")
    tauri = json.loads(
        (root / "apps/product-client/src-tauri/tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    package = json.loads(
        (root / "apps/product-client/package.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "python": toml_section_version(
            pyproject_text,
            "project",
            source="pyproject.toml",
        ),
        "cargo": toml_section_version(
            cargo_text,
            "package",
            source="Cargo.toml",
        ),
        "tauri": str(tauri["version"]),
        "package": str(package["version"]),
    }


def verify_product_versions(root: Path) -> dict[str, str]:
    versions = product_versions(root)
    if len(set(versions.values())) != 1:
        raise ValueError(
            "product version mismatch: "
            + ", ".join(
                f"{name}={value}"
                for name, value in sorted(versions.items())
            )
        )
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    versions = verify_product_versions(Path(args.root).resolve())
    print(json.dumps(versions, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
