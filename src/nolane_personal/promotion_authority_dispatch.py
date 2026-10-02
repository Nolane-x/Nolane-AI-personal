from __future__ import annotations

from typing import Any

from .promotion_authority import (
    AUTH_SCHEMA as L35_AUTH_SCHEMA,
    verify_promotion_authorization,
)
from .unified_promotion_authority import (
    AUTH_SCHEMA as L40_AUTH_SCHEMA,
    verify_unified_promotion_authorization,
)


def verify_any_promotion_authorization(
    authorization: dict[str, Any],
    *,
    now: str | None = None,
    require_authorized: bool = True,
    check_expiry: bool = True,
) -> dict[str, Any]:
    schema = authorization.get("schema")
    if schema == L35_AUTH_SCHEMA:
        return verify_promotion_authorization(
            authorization,
            now=now,
            require_authorized=require_authorized,
            check_expiry=check_expiry,
        )
    if schema == L40_AUTH_SCHEMA:
        return verify_unified_promotion_authorization(
            authorization,
            now=now,
            require_authorized=require_authorized,
            check_expiry=check_expiry,
        )
    raise ValueError("unsupported promotion authorization schema")


def authorization_chain_sha256(
    authorization: dict[str, Any],
) -> str:
    schema = authorization.get("schema")
    if schema == L35_AUTH_SCHEMA:
        return str(authorization["multicycle_chain_sha256"])
    if schema == L40_AUTH_SCHEMA:
        return str(authorization["unified_chain_sha256"])
    raise ValueError("unsupported promotion authorization schema")


def authorization_kind(
    authorization: dict[str, Any],
) -> str:
    schema = authorization.get("schema")
    if schema == L35_AUTH_SCHEMA:
        return "L35_BOUNDARY_CHAIN"
    if schema == L40_AUTH_SCHEMA:
        return "L40_UNIFIED_MODEL_CHAIN"
    raise ValueError("unsupported promotion authorization schema")
