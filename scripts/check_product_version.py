from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def product_versions(root: Path) -> dict[str, str]:
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*    tauri = json.loads(
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
        "python": match.group(1),
        "tauri": str(tauri["version"]),
        "package": str(package["version"]),
    }


def verify_product_versions(root: Path) -> dict[str, str]:
    versions = product_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
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
,
        pyproject_text,
    )
    if match is None:
        raise ValueError("project version not found in pyproject.toml")
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
        "python": str(pyproject["project"]["version"]),
        "tauri": str(tauri["version"]),
        "package": str(package["version"]),
    }


def verify_product_versions(root: Path) -> dict[str, str]:
    versions = product_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
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
