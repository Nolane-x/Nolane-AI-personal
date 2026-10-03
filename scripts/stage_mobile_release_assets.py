from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from nolane_personal.mobile_factorized import (
    verify_mobile_factorized_package,
)
from nolane_personal.mobile_persistent_state import (
    read_persistent_mobile_state,
)
from nolane_personal.mobile_prompt_contract import (
    sha256_file,
    verify_product_prompt_contract,
)
from nolane_personal.promotion_ceremony import (
    verify_promotion_ceremony_receipt,
)


SCHEMA = "NOLANE-V057-AUTHORIZED-LOCALMOBILE-BUNDLE-V1"
AUTHORITY = "L36_COMPLETE_PROMOTION_CEREMONY"


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


def stage_mobile_release_assets(
    *,
    mobile_package_dir: Path,
    tokenizer_dir: Path,
    prompt_contract_path: Path,
    bootstrap_state_path: Path,
    ceremony_path: Path,
    resources: Path,
) -> dict[str, object]:
    if not ceremony_path.is_file():
        raise SystemExit(f"promotion ceremony missing: {ceremony_path}")
    ceremony = json.loads(ceremony_path.read_text(encoding="utf-8"))
    verify_promotion_ceremony_receipt(
        ceremony,
        require_complete=True,
    )
    checkpoint_sha256 = str(
        ceremony["candidate_checkpoint_sha256"]
    ).lower()

    package = verify_mobile_factorized_package(
        mobile_package_dir,
        expected_source_checkpoint_sha256=checkpoint_sha256,
    )
    package_contract = json.loads(
        (
            mobile_package_dir
            / str(package["contract_filename"])
        ).read_text(encoding="utf-8")
    )
    latent_dim = int(package_contract["latent_dim"])

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    missing = [
        str(path)
        for path in (tokenizer_json, tokenizer_config)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "mobile tokenizer release assets missing: "
            + ", ".join(missing)
        )

    if not prompt_contract_path.is_file():
        raise SystemExit(
            f"mobile prompt contract missing: {prompt_contract_path}"
        )
    prompt_contract = json.loads(
        prompt_contract_path.read_text(encoding="utf-8")
    )
    verify_product_prompt_contract(
        prompt_contract,
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=tokenizer_config,
    )

    read_persistent_mobile_state(
        bootstrap_state_path,
        expected_source_checkpoint_sha256=checkpoint_sha256,
        expected_latent_dim=latent_dim,
    )

    resources.mkdir(parents=True, exist_ok=True)
    clear_directory(resources)
    package_dest = resources / "package"
    copy_tree(mobile_package_dir, package_dest)
    shutil.copy2(tokenizer_json, resources / "tokenizer.json")
    shutil.copy2(
        tokenizer_config,
        resources / "tokenizer_config.json",
    )
    shutil.copy2(
        prompt_contract_path,
        resources / "prompt-contract.json",
    )
    shutil.copy2(
        bootstrap_state_path,
        resources / "bootstrap-state.json",
    )
    shutil.copy2(
        ceremony_path,
        resources / "promotion-ceremony.json",
    )

    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_checkpoint_sha256": checkpoint_sha256,
        "prompt_contract_file_sha256": sha256_file(
            resources / "prompt-contract.json"
        ),
        "mobile_package_manifest_sha256": sha256_file(
            package_dest / "manifest.json"
        ),
        "tokenizer_json_sha256": sha256_file(
            resources / "tokenizer.json"
        ),
        "tokenizer_config_json_sha256": sha256_file(
            resources / "tokenizer_config.json"
        ),
        "bootstrap_state_sha256": sha256_file(
            resources / "bootstrap-state.json"
        ),
        "promotion_ceremony_sha256": ceremony[
            "ceremony_sha256"
        ],
        "promotion_authorization_sha256": ceremony[
            "authorization_sha256"
        ],
        "promotion_ceremony_file_sha256": sha256_file(
            resources / "promotion-ceremony.json"
        ),
        "mobile_package_internal_manifest_sha256": package[
            "manifest_sha256"
        ],
        "mobile_weights_sha256": package["weights_sha256"],
        "mobile_contract_sha256": package["contract_sha256"],
        "tokenizer_contract_sha256": prompt_contract[
            "contract_sha256"
        ],
        "release_claims": {
            "l36_complete": True,
            "checkpoint_bound": True,
            "python_required_on_android": False,
            "qwen_model_object_required": False,
            "device_court_complete": False,
        },
    }
    (resources / "localmobile-manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile-package-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt-contract", required=True)
    parser.add_argument("--bootstrap-state", required=True)
    parser.add_argument("--ceremony", required=True)
    parser.add_argument(
        "--resources",
        default="apps/product-client/src-tauri/resources/mobile",
    )
    args = parser.parse_args()

    manifest = stage_mobile_release_assets(
        mobile_package_dir=Path(args.mobile_package_dir),
        tokenizer_dir=Path(args.tokenizer_dir),
        prompt_contract_path=Path(args.prompt_contract),
        bootstrap_state_path=Path(args.bootstrap_state),
        ceremony_path=Path(args.ceremony),
        resources=Path(args.resources),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
