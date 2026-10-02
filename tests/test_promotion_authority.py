import hashlib
import json

import pytest

from nolane_personal.continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
)
from nolane_personal.long_horizon_retention import (
    LongHorizonRetentionPolicy,
    assess_long_horizon_retention,
)
from nolane_personal.multicycle_continual import (
    assess_multicycle_continual_chain,
)
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol
from nolane_personal.promotion_authority import (
    PromotionAuthorizationPolicy,
    create_operator_promotion_request,
    decide_promotion_authorization,
    verify_promotion_authorization,
)
from nolane_personal.store import payload_digest
from nolane_personal.transactional_registry import (
    CheckpointRegistry,
    sha256_file,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def protocol():
    examples = [
        PersonalizationExample(
            prompt=f"prompt {i}",
            target=f"target {i}",
            language="vi" if i % 2 == 0 else "en",
        )
        for i in range(8)
    ]
    groups = [
        digest(name)
        for name in ("a", "a", "b", "c", "d", "e", "f", "g")
    ]
    return build_personalization_protocol(
        examples,
        dataset_sha256=digest("dataset"),
        source_group_sha256=groups,
    )


def continual(label, before, after):
    receipt = assess_continual_learning(
        pre_update_checkpoint_sha256=before,
        post_update_checkpoint_sha256=after,
        retention_group_sha256=[
            digest(f"{label}-old-a"),
            digest(f"{label}-old-b"),
        ],
        retention_before_values=[1.0, 1.1],
        retention_after_values=[1.005, 1.105],
        adaptation_group_sha256=[
            digest(f"{label}-new-a"),
            digest(f"{label}-new-b"),
        ],
        adaptation_before_values=[1.4, 1.3],
        adaptation_after_values=[1.2, 1.1],
        policy=ContinualLearningPolicy(
            min_retention_groups=2,
            min_adaptation_groups=2,
            max_worst_retention_regression=0.02,
            max_mean_retention_regression=0.02,
            min_mean_adaptation_gain=0.10,
            max_worst_adaptation_regression=0.0,
        ),
    )
    assert receipt["status"] == "PASS"
    return receipt


def run_receipt(
    label,
    *,
    parent_sha,
    artifact_sha,
    boundary_before,
    boundary_after,
):
    cortex = digest("stable-cortex")
    l30 = continual(label, boundary_before, boundary_after)
    lineage = {
        "schema": "NOLANE-L31-CONTINUAL-UPDATE-LINEAGE-V1",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "retention_dataset_sha256": digest(f"{label}-retention-dataset"),
        "retention_protocol_sha256": digest(f"{label}-retention-protocol"),
        "retention_quality_court_sha256": digest(f"{label}-retention-quality"),
        "adaptation_dataset_sha256": digest(f"{label}-adaptation-dataset"),
        "adaptation_protocol_sha256": digest(f"{label}-adaptation-protocol"),
        "adaptation_quality_court_sha256": digest(f"{label}-adaptation-quality"),
    }
    lineage["lineage_sha256"] = payload_digest(lineage)
    training = {
        "schema": "NOLANE-L31-FACTORIZED-CONTINUAL-UPDATE-V1",
        "authority": "CONTINUAL_FACTORIZED_UPDATE_CANDIDATE_ONLY",
        "adaptation_train_examples": 4,
        "retention_rehearsal_examples": 4,
        "retention_eval_examples": 2,
        "adaptation_eval_examples": 2,
        "epochs": 1,
        "optimizer_steps": 4,
        "adaptation_nll_before": 1.35,
        "adaptation_nll_after": 1.15,
        "retention_nll_before": 1.05,
        "retention_nll_after": 1.055,
        "candidate_boundary_digest_before": boundary_before,
        "candidate_boundary_digest_after": boundary_after,
        "candidate_boundary_changed": True,
        "reference_boundary_digest_before": boundary_before,
        "reference_boundary_digest_after": boundary_before,
        "reference_boundary_unchanged": True,
        "candidate_cortex_digest_before": cortex,
        "candidate_cortex_digest_after": cortex,
        "candidate_cortex_unchanged": True,
        "reference_cortex_digest_before": cortex,
        "reference_cortex_digest_after": cortex,
        "reference_cortex_unchanged": True,
        "candidate_boundary_gradients_seen": 4,
        "candidate_cortex_gradients_seen": 0,
        "config": {},
        "continual_learning": l30,
        "lineage": lineage,
    }
    manifest = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": artifact_sha,
        "boundary_state_digest": boundary_after,
        "cortex_state_digest": cortex,
        "dataset_fingerprint": lineage["lineage_sha256"],
        "training_receipt": training,
    }
    return {
        "schema": "NOLANE-L31-CONTINUAL-FACTORIZED-UPDATE-RUN-V1",
        "authority": "CONTINUAL_UPDATE_EVIDENCE_ONLY_UNPROMOTED",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "training": training,
        "artifact": manifest,
    }


def fixed_horizon(initial_sha, final_sha):
    frozen = protocol()
    rows = frozen["splits"]["test"]
    base = [1.0 + 0.1 * i for i in range(len(rows))]
    receipt = assess_long_horizon_retention(
        frozen,
        initial_checkpoint_sha256=initial_sha,
        final_checkpoint_sha256=final_sha,
        initial_values=base,
        final_values=[value + 0.005 for value in base],
        policy=LongHorizonRetentionPolicy(
            max_overall_regression=0.01,
            max_worst_group_regression=0.02,
        ),
    )
    assert receipt["status"] == "PASS"
    return receipt


def initial_bundle(tmp_path):
    bundle = tmp_path / "parent-bundle"
    bundle.mkdir()
    checkpoint = bundle / "factorized-nolane.pt"
    checkpoint.write_bytes(b"production-parent")
    parent_sha = sha256_file(checkpoint)
    manifest = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": parent_sha,
        "boundary_state_digest": digest("b0"),
        "cortex_state_digest": digest("stable-cortex"),
        "dataset_fingerprint": digest("parent-dataset"),
        "training_receipt": None,
    }
    write_json(bundle / "factorized-nolane-manifest.json", manifest)
    return bundle, parent_sha


def evidence(tmp_path):
    parent_bundle, parent_sha = initial_bundle(tmp_path)
    b0 = digest("b0")
    b1 = digest("b1")
    b2 = digest("b2")
    artifact1 = digest("offline-artifact-1")

    final_bundle = tmp_path / "final-bundle"
    final_bundle.mkdir()
    final_checkpoint = final_bundle / "factorized-nolane.pt"
    final_checkpoint.write_bytes(b"offline-final-candidate")
    artifact2 = sha256_file(final_checkpoint)

    cycle1 = run_receipt(
        "cycle-1",
        parent_sha=parent_sha,
        artifact_sha=artifact1,
        boundary_before=b0,
        boundary_after=b1,
    )
    cycle2 = run_receipt(
        "cycle-2",
        parent_sha=artifact1,
        artifact_sha=artifact2,
        boundary_before=b1,
        boundary_after=b2,
    )
    write_json(
        final_bundle / "factorized-nolane-manifest.json",
        cycle2["artifact"],
    )
    write_json(final_bundle / "l31-run-receipt.json", cycle2)

    horizon = fixed_horizon(parent_sha, artifact2)
    cycles = [cycle1, cycle2]
    chain = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert chain["status"] == "PASS"
    return {
        "parent_bundle": parent_bundle,
        "parent_sha": parent_sha,
        "final_bundle": final_bundle,
        "final_sha": artifact2,
        "cycles": cycles,
        "horizon": horizon,
        "chain": chain,
    }


def request(ev, *, approved=True, candidate=None):
    return create_operator_promotion_request(
        active_parent_checkpoint_sha256=ev["parent_sha"],
        candidate_checkpoint_sha256=candidate or ev["final_sha"],
        multicycle_chain_sha256=ev["chain"]["chain_sha256"],
        long_horizon_retention_court_sha256=ev["horizon"]["court_sha256"],
        approved=approved,
        nonce="0123456789abcdef-promote",
        created_at="2026-10-02T00:00:00+00:00",
    )


def authorization(ev, *, approved=True, ttl=3600):
    return decide_promotion_authorization(
        cycles=ev["cycles"],
        long_horizon_retention=ev["horizon"],
        multicycle_chain=ev["chain"],
        operator_request=request(ev, approved=approved),
        policy=PromotionAuthorizationPolicy(
            min_cycles=2,
            authorization_ttl_seconds=ttl,
        ),
        now="2026-10-02T00:00:01+00:00",
    )


def test_valid_l32_chain_plus_operator_intent_authorizes_final_candidate(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev)
    assert auth["status"] == "AUTHORIZED"
    assert auth["active_parent_checkpoint_sha256"] == ev["parent_sha"]
    assert auth["candidate_checkpoint_sha256"] == ev["final_sha"]
    verify_promotion_authorization(
        auth,
        now="2026-10-02T00:10:00+00:00",
    )


def test_operator_denial_blocks_authorization(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev, approved=False)
    assert auth["status"] == "BLOCKED"
    assert "operator_did_not_approve" in auth["reasons"]
    with pytest.raises(ValueError, match="not AUTHORIZED"):
        verify_promotion_authorization(
            auth,
            now="2026-10-02T00:10:00+00:00",
        )


def test_operator_candidate_mismatch_blocks(tmp_path):
    ev = evidence(tmp_path)
    req = request(ev, candidate=digest("wrong-candidate"))
    auth = decide_promotion_authorization(
        cycles=ev["cycles"],
        long_horizon_retention=ev["horizon"],
        multicycle_chain=ev["chain"],
        operator_request=req,
        now="2026-10-02T00:00:01+00:00",
    )
    assert auth["status"] == "BLOCKED"
    assert "operator_candidate_checkpoint_mismatch" in auth["reasons"]


def test_authorization_tamper_and_expiry_fail_closed(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev, ttl=10)
    tampered = dict(auth)
    tampered["candidate_checkpoint_sha256"] = digest("tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_promotion_authorization(
            tampered,
            now="2026-10-02T00:00:05+00:00",
        )
    with pytest.raises(ValueError, match="expired"):
        verify_promotion_authorization(
            auth,
            now="2026-10-02T00:00:20+00:00",
        )


def test_chain_is_recomputed_not_trusted_by_digest_alone(tmp_path):
    ev = evidence(tmp_path)
    # Each cycle remains individually valid, but their order no longer matches
    # the supplied L32 chain evidence.
    changed_cycles = [ev["cycles"][1], ev["cycles"][0]]

    with pytest.raises(ValueError):
        decide_promotion_authorization(
            cycles=changed_cycles,
            long_horizon_retention=ev["horizon"],
            multicycle_chain=ev["chain"],
            operator_request=request(ev),
            now="2026-10-02T00:00:01+00:00",
        )


def test_authorized_registry_can_jump_from_chain_start_to_final_artifact(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(ev["parent_bundle"])

    # The final L31 candidate's immediate parent is offline artifact 1,
    # not the production-active checkpoint. Direct L33 insertion is rejected.
    with pytest.raises(ValueError, match="parent does not match active"):
        registry.begin_l31_update(ev["final_bundle"])

    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        auth,
        now="2026-10-02T00:00:02+00:00",
    )
    registry.verify_update(
        tx,
        now="2026-10-02T00:00:03+00:00",
    )
    pointer = registry.commit_update(
        tx,
        now="2026-10-02T00:00:04+00:00",
    )
    assert pointer["checkpoint_sha256"] == ev["final_sha"]
    audit = registry.verify_registry()
    assert audit["status"] == "PASS"
    assert audit["promotion_authorizations"] == 1


def test_promotion_authorization_is_one_time_at_transaction_begin(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(ev["parent_bundle"])
    registry.begin_authorized_update(
        ev["final_bundle"],
        auth,
        now="2026-10-02T00:00:02+00:00",
    )
    with pytest.raises(RuntimeError, match="already been used"):
        registry.begin_authorized_update(
            ev["final_bundle"],
            auth,
            now="2026-10-02T00:00:03+00:00",
        )


def test_authorization_expiring_between_verify_and_commit_blocks_swap(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev, ttl=10)
    registry = CheckpointRegistry(tmp_path / "registry")
    p0 = registry.initialize(ev["parent_bundle"])
    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        auth,
        now="2026-10-02T00:00:02+00:00",
    )
    registry.verify_update(
        tx,
        now="2026-10-02T00:00:05+00:00",
    )
    with pytest.raises(ValueError, match="expired"):
        registry.commit_update(
            tx,
            now="2026-10-02T00:00:20+00:00",
        )
    assert registry.active_pointer() == p0


def test_registry_audit_detects_authorization_file_tamper(tmp_path):
    ev = evidence(tmp_path)
    auth = authorization(ev)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(ev["parent_bundle"])
    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        auth,
        now="2026-10-02T00:00:02+00:00",
    )
    path = registry._tx_dir(tx) / "promotion-authorization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cycles"] = 999
    write_json(path, payload)
    with pytest.raises(ValueError, match="authorization digest mismatch"):
        registry.verify_registry()
