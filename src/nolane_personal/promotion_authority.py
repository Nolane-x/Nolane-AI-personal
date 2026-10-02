from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .long_horizon_retention import verify_long_horizon_retention_digest
from .multicycle_continual import (
    MultiCycleContinualPolicy,
    assess_multicycle_continual_chain,
    verify_multicycle_chain_digest,
)
from .state import utc_now_iso
from .store import payload_digest


REQUEST_SCHEMA = "NOLANE-L35-OPERATOR-PROMOTION-REQUEST-V1"
AUTH_SCHEMA = "NOLANE-L35-EVIDENCE-BOUND-PROMOTION-AUTHORIZATION-V1"
AUTHORITY = "EXPLICIT_EVIDENCE_BOUND_PROMOTION_AUTHORITY"


@dataclass(slots=True)
class PromotionAuthorizationPolicy:
    min_cycles: int = 2
    authorization_ttl_seconds: int = 3600
    require_unique_adaptation_protocols: bool = True
    require_operator_approval: bool = True

    def validate(self) -> None:
        if self.min_cycles < 2:
            raise ValueError("min_cycles must be >=2")
        if self.authorization_ttl_seconds < 1:
            raise ValueError("authorization_ttl_seconds must be positive")
        if not self.require_unique_adaptation_protocols:
            raise ValueError(
                "promotion authority requires unique adaptation protocols"
            )
        if not self.require_operator_approval:
            raise ValueError("promotion authority requires operator approval")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("promotion timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a sha256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 hex string") from exc
    return value


def create_operator_promotion_request(
    *,
    active_parent_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    multicycle_chain_sha256: str,
    long_horizon_retention_court_sha256: str,
    approved: bool,
    nonce: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    parent = _validate_sha256(
        active_parent_checkpoint_sha256,
        name="active_parent_checkpoint_sha256",
    )
    candidate = _validate_sha256(
        candidate_checkpoint_sha256,
        name="candidate_checkpoint_sha256",
    )
    chain = _validate_sha256(
        multicycle_chain_sha256,
        name="multicycle_chain_sha256",
    )
    horizon = _validate_sha256(
        long_horizon_retention_court_sha256,
        name="long_horizon_retention_court_sha256",
    )
    nonce_value = str(nonce)
    if len(nonce_value) < 16:
        raise ValueError("promotion request nonce must be at least 16 characters")
    timestamp = created_at or utc_now_iso()
    _parse_time(timestamp)
    request = {
        "schema": REQUEST_SCHEMA,
        "authority": "EXPLICIT_OPERATOR_INTENT_NO_PROMOTION_BY_ITSELF",
        "approved": bool(approved),
        "active_parent_checkpoint_sha256": parent,
        "candidate_checkpoint_sha256": candidate,
        "multicycle_chain_sha256": chain,
        "long_horizon_retention_court_sha256": horizon,
        "operator_nonce_sha256": _sha256_text(nonce_value),
        "created_at": timestamp,
    }
    request["request_sha256"] = payload_digest(request)
    return request


def verify_operator_promotion_request(
    request: dict[str, Any],
) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported promotion request schema")
    supplied = request.get("request_sha256")
    body = dict(request)
    body.pop("request_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("promotion request digest mismatch")
    for key in (
        "active_parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "multicycle_chain_sha256",
        "long_horizon_retention_court_sha256",
        "operator_nonce_sha256",
        "request_sha256",
    ):
        _validate_sha256(request.get(key), name=key)
    _parse_time(str(request.get("created_at")))
    if not isinstance(request.get("approved"), bool):
        raise ValueError("promotion request approved must be boolean")
    return request


def decide_promotion_authorization(
    *,
    cycles: list[dict[str, Any]],
    long_horizon_retention: dict[str, Any],
    multicycle_chain: dict[str, Any],
    operator_request: dict[str, Any],
    policy: PromotionAuthorizationPolicy | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    policy = policy or PromotionAuthorizationPolicy()
    policy.validate()
    verify_operator_promotion_request(operator_request)
    verify_long_horizon_retention_digest(long_horizon_retention)
    verify_multicycle_chain_digest(multicycle_chain)

    chain_policy = MultiCycleContinualPolicy(
        **dict(multicycle_chain.get("policy", {}))
    )
    if not chain_policy.require_unique_adaptation_protocols:
        raise ValueError(
            "promotion cannot use chain evidence that allowed adaptation reuse"
        )
    recomputed = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=long_horizon_retention,
        policy=chain_policy,
    )
    if recomputed != multicycle_chain:
        raise ValueError("multicycle chain does not match supplied cycle evidence")

    reasons: list[str] = []
    if multicycle_chain.get("status") != "PASS":
        reasons.append("multicycle_chain_not_pass")
    if long_horizon_retention.get("status") != "PASS":
        reasons.append("long_horizon_retention_not_pass")
    if int(multicycle_chain.get("cycles", 0)) < policy.min_cycles:
        reasons.append("insufficient_multicycle_evidence")
    if not operator_request.get("approved"):
        reasons.append("operator_did_not_approve")

    first_parent = _validate_sha256(
        multicycle_chain.get("first_parent_checkpoint_sha256"),
        name="first_parent_checkpoint_sha256",
    )
    final_candidate = _validate_sha256(
        multicycle_chain.get("final_artifact_checkpoint_sha256"),
        name="final_artifact_checkpoint_sha256",
    )
    chain_sha = _validate_sha256(
        multicycle_chain.get("chain_sha256"),
        name="chain_sha256",
    )
    horizon_sha = _validate_sha256(
        long_horizon_retention.get("court_sha256"),
        name="court_sha256",
    )

    if (
        operator_request["active_parent_checkpoint_sha256"]
        != first_parent
    ):
        reasons.append("operator_parent_checkpoint_mismatch")
    if operator_request["candidate_checkpoint_sha256"] != final_candidate:
        reasons.append("operator_candidate_checkpoint_mismatch")
    if operator_request["multicycle_chain_sha256"] != chain_sha:
        reasons.append("operator_chain_mismatch")
    if (
        operator_request["long_horizon_retention_court_sha256"]
        != horizon_sha
    ):
        reasons.append("operator_long_horizon_mismatch")

    issued = _parse_time(now or utc_now_iso())
    expires = issued + timedelta(
        seconds=int(policy.authorization_ttl_seconds)
    )
    receipt = {
        "schema": AUTH_SCHEMA,
        "authority": AUTHORITY,
        "status": "AUTHORIZED" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "policy": asdict(policy),
        "active_parent_checkpoint_sha256": first_parent,
        "candidate_checkpoint_sha256": final_candidate,
        "multicycle_chain_sha256": chain_sha,
        "long_horizon_retention_court_sha256": horizon_sha,
        "cycles": int(multicycle_chain.get("cycles", 0)),
        "operator_request_sha256": operator_request["request_sha256"],
        "operator_nonce_sha256": operator_request["operator_nonce_sha256"],
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "privacy": {
            "contains_raw_prompt_target": False,
            "contains_raw_operator_nonce": False,
        },
    }
    receipt["authorization_sha256"] = payload_digest(receipt)
    return receipt


def verify_promotion_authorization(
    authorization: dict[str, Any],
    *,
    now: str | None = None,
    require_authorized: bool = True,
    check_expiry: bool = True,
) -> dict[str, Any]:
    if authorization.get("schema") != AUTH_SCHEMA:
        raise ValueError("unsupported promotion authorization schema")
    supplied = authorization.get("authorization_sha256")
    body = dict(authorization)
    body.pop("authorization_sha256", None)
    if payload_digest(body) != supplied:
        raise ValueError("promotion authorization digest mismatch")
    if authorization.get("authority") != AUTHORITY:
        raise ValueError("promotion authorization authority mismatch")
    for key in (
        "active_parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "multicycle_chain_sha256",
        "long_horizon_retention_court_sha256",
        "operator_request_sha256",
        "operator_nonce_sha256",
        "authorization_sha256",
    ):
        _validate_sha256(authorization.get(key), name=key)
    issued = _parse_time(str(authorization.get("issued_at")))
    expires = _parse_time(str(authorization.get("expires_at")))
    if expires <= issued:
        raise ValueError("promotion authorization expiry invalid")
    if check_expiry:
        current = _parse_time(now or utc_now_iso())
        if current > expires:
            raise ValueError("promotion authorization expired")
    if require_authorized and authorization.get("status") != "AUTHORIZED":
        raise ValueError("promotion authorization is not AUTHORIZED")
    return authorization
