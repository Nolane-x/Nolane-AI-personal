from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .factorized_artifact import load_factorized_model
from .mobile_factorized import (
    export_mobile_factorized_package,
    verify_mobile_factorized_package,
)
from .mobile_persistent_state import (
    build_persistent_mobile_state,
    read_persistent_mobile_state,
    write_persistent_mobile_state,
)
from .mobile_prompt_contract import (
    freeze_product_prompt_contract,
    sha256_file,
    verify_product_prompt_contract,
    write_product_prompt_contract,
)
from .product_profile import ProductProfile
from .promotion_ceremony import verify_promotion_ceremony_receipt
from .state import LivingState
from .store import canonical_json, payload_digest


RELEASE_SCHEMA = "NOLANE-V057-LOCALMOBILE-RELEASE-BUNDLE-V1"
RELEASE_AUTHORITY = "L36_COMPLETE_PROMOTION_BOUND_LOCALMOBILE"
BOOTSTRAP_IDENTITY = "NOLANE-LOCALMOBILE-BOOTSTRAP-IDENTITY-REPLACED"


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return text


def _latent_dim_from_checkpoint(checkpoint: Path) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LocalMobile release staging requires torch") from exc

    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    cortex = dict(payload.get("cortex_config", {}))
    latent_dim = int(cortex.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError("factorized checkpoint is missing latent_dim")
    return latent_dim


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _build_release_manifest(
    root: Path,
    *,
    source_checkpoint_sha256: str,
    ceremony: dict[str, Any],
    mobile_package_manifest: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "authority": RELEASE_AUTHORITY,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "promotion_ceremony_file": "promotion-ceremony.json",
        "promotion_ceremony_file_sha256": sha256_file(
            root / "promotion-ceremony.json"
        ),
        "promotion_ceremony_sha256": ceremony["ceremony_sha256"],
        "promotion_authorization_sha256": ceremony["authorization_sha256"],
        "mobile_package_manifest_sha256": mobile_package_manifest[
            "manifest_sha256"
        ],
        "tokenizer_json_sha256": sha256_file(root / "tokenizer.json"),
        "tokenizer_config_json_sha256": sha256_file(
            root / "tokenizer_config.json"
        ),
        "prompt_contract_file_sha256": sha256_file(
            root / "prompt-contract.json"
        ),
        "bootstrap_state_file_sha256": sha256_file(
            root / "bootstrap-state.json"
        ),
        "release_claims": {
            "l36_complete_required": True,
            "python_required_on_android": False,
            "loopback_http_required_on_android": False,
            "device_court_complete": False,
        },
    }
    body["release_manifest_sha256"] = payload_digest(body)
    return body


def verify_localmobile_release_bundle(
    root: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "localmobile-manifest.json"
    manifest = _load_json(manifest_path)

    if manifest.get("schema") != RELEASE_SCHEMA:
        raise ValueError("unsupported LocalMobile release schema")
    if manifest.get("authority") != RELEASE_AUTHORITY:
        raise ValueError("LocalMobile release authority mismatch")

    supplied_manifest_sha = _require_sha256(
        manifest.get("release_manifest_sha256"),
        name="release_manifest_sha256",
    )
    body = dict(manifest)
    body.pop("release_manifest_sha256", None)
    if payload_digest(body) != supplied_manifest_sha:
        raise ValueError("LocalMobile release manifest digest mismatch")

    source_sha = _require_sha256(
        manifest.get("source_checkpoint_sha256"),
        name="source_checkpoint_sha256",
    )
    for key in (
        "promotion_ceremony_file_sha256",
        "promotion_ceremony_sha256",
        "promotion_authorization_sha256",
        "mobile_package_manifest_sha256",
        "tokenizer_json_sha256",
        "tokenizer_config_json_sha256",
        "prompt_contract_file_sha256",
        "bootstrap_state_file_sha256",
    ):
        _require_sha256(manifest.get(key), name=key)

    ceremony_path = root / str(manifest["promotion_ceremony_file"])
    if sha256_file(ceremony_path) != manifest["promotion_ceremony_file_sha256"]:
        raise ValueError("LocalMobile ceremony file SHA-256 mismatch")
    ceremony = _load_json(ceremony_path)
    verify_promotion_ceremony_receipt(ceremony, require_complete=True)
    if ceremony["candidate_checkpoint_sha256"] != source_sha:
        raise ValueError("LocalMobile ceremony checkpoint mismatch")
    if ceremony["ceremony_sha256"] != manifest["promotion_ceremony_sha256"]:
        raise ValueError("LocalMobile ceremony semantic digest mismatch")
    if (
        ceremony["authorization_sha256"]
        != manifest["promotion_authorization_sha256"]
    ):
        raise ValueError("LocalMobile promotion authorization mismatch")

    mobile_package = verify_mobile_factorized_package(
        root / "package",
        expected_source_checkpoint_sha256=source_sha,
    )
    if (
        mobile_package["manifest_sha256"]
        != manifest["mobile_package_manifest_sha256"]
    ):
        raise ValueError("LocalMobile mobile-package manifest mismatch")

    tokenizer_json = root / "tokenizer.json"
    tokenizer_config = root / "tokenizer_config.json"
    if sha256_file(tokenizer_json) != manifest["tokenizer_json_sha256"]:
        raise ValueError("LocalMobile tokenizer.json SHA-256 mismatch")
    if (
        sha256_file(tokenizer_config)
        != manifest["tokenizer_config_json_sha256"]
    ):
        raise ValueError(
            "LocalMobile tokenizer_config.json SHA-256 mismatch"
        )

    prompt_contract_path = root / "prompt-contract.json"
    if (
        sha256_file(prompt_contract_path)
        != manifest["prompt_contract_file_sha256"]
    ):
        raise ValueError("LocalMobile prompt contract file SHA-256 mismatch")
    prompt_contract = _load_json(prompt_contract_path)
    verify_product_prompt_contract(
        prompt_contract,
        tokenizer_json=tokenizer_json,
        tokenizer_config_json=tokenizer_config,
    )

    bootstrap_path = root / "bootstrap-state.json"
    if (
        sha256_file(bootstrap_path)
        != manifest["bootstrap_state_file_sha256"]
    ):
        raise ValueError("LocalMobile bootstrap state file SHA-256 mismatch")
    contract = _load_json(root / "package" / "contract.json")
    latent_dim = int(contract.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError("LocalMobile package latent_dim missing")
    bootstrap = read_persistent_mobile_state(
        bootstrap_path,
        expected_source_checkpoint_sha256=source_sha,
        expected_latent_dim=latent_dim,
    )
    runtime_state = dict(bootstrap["state"])
    if runtime_state.get("identity_id") != BOOTSTRAP_IDENTITY:
        raise ValueError("LocalMobile bootstrap identity sentinel mismatch")
    if runtime_state.get("open_threads") != []:
        raise ValueError("LocalMobile bootstrap must not ship open threads")
    if bootstrap.get("memories") != []:
        raise ValueError("LocalMobile bootstrap must not ship user memories")

    claims = manifest.get("release_claims")
    if claims != {
        "l36_complete_required": True,
        "python_required_on_android": False,
        "loopback_http_required_on_android": False,
        "device_court_complete": False,
    }:
        raise ValueError("LocalMobile release claims mismatch")
    return manifest


def stage_localmobile_release_bundle(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    tokenizer_dir: str | Path,
    ceremony_path: str | Path,
    resources: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    ceremony_path = Path(ceremony_path).resolve()
    resources = Path(resources).resolve()

    if checkpoint.is_dir():
        checkpoint = checkpoint / "factorized-nolane.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(tokenizer_dir)
    if not ceremony_path.is_file():
        raise FileNotFoundError(ceremony_path)

    expected_sha = _require_sha256(
        str(checkpoint_sha256).strip().lower(),
        name="checkpoint_sha256",
    )
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            "LocalMobile release checkpoint SHA-256 mismatch: "
            f"expected={expected_sha} actual={actual_sha}"
        )

    ceremony = _load_json(ceremony_path)
    verify_promotion_ceremony_receipt(ceremony, require_complete=True)
    if ceremony["candidate_checkpoint_sha256"] != actual_sha:
        raise ValueError("LocalMobile ceremony checkpoint mismatch")

    tokenizer_json_source = tokenizer_dir / "tokenizer.json"
    tokenizer_config_source = tokenizer_dir / "tokenizer_config.json"
    if not tokenizer_json_source.is_file():
        raise FileNotFoundError(tokenizer_json_source)
    if not tokenizer_config_source.is_file():
        raise FileNotFoundError(tokenizer_config_source)

    latent_dim = _latent_dim_from_checkpoint(checkpoint)
    model, metadata = load_factorized_model(
        checkpoint,
        [0.0] * latent_dim,
        device="cpu",
    )
    if str(metadata["checkpoint_sha256"]) != actual_sha:
        raise RuntimeError("loaded factorized checkpoint digest drift")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LocalMobile release staging requires transformers"
        ) from exc

    parent = resources.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".localmobile-release-",
            dir=parent,
        )
    )
    try:
        mobile_package = export_mobile_factorized_package(
            model,
            temporary / "package",
            source_checkpoint_sha256=actual_sha,
        )

        shutil.copy2(tokenizer_json_source, temporary / "tokenizer.json")
        shutil.copy2(
            tokenizer_config_source,
            temporary / "tokenizer_config.json",
        )
        shutil.copy2(
            ceremony_path,
            temporary / "promotion-ceremony.json",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        prompt_contract = freeze_product_prompt_contract(
            tokenizer,
            tokenizer_json=temporary / "tokenizer.json",
            tokenizer_config_json=temporary / "tokenizer_config.json",
        )
        write_product_prompt_contract(
            prompt_contract,
            temporary / "prompt-contract.json",
        )

        profile = ProductProfile()
        living_state = LivingState(identity_id=BOOTSTRAP_IDENTITY)
        living_state.open_threads = []
        persistent_state = build_persistent_mobile_state(
            source_checkpoint_sha256=actual_sha,
            latent=[0.0] * latent_dim,
            profile=profile,
            state=living_state,
            memories=[],
        )
        write_persistent_mobile_state(
            temporary / "bootstrap-state.json",
            persistent_state,
            expected_source_checkpoint_sha256=actual_sha,
            expected_latent_dim=latent_dim,
        )

        manifest = _build_release_manifest(
            temporary,
            source_checkpoint_sha256=actual_sha,
            ceremony=ceremony,
            mobile_package_manifest=mobile_package,
        )
        (temporary / "localmobile-manifest.json").write_text(
            canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        verify_localmobile_release_bundle(temporary)

        if resources.exists():
            shutil.rmtree(resources)
        shutil.move(str(temporary), str(resources))
        return verify_localmobile_release_bundle(resources)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
