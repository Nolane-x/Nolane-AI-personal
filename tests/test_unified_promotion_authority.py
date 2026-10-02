import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nolane_personal.continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
)
from nolane_personal.long_horizon_retention import (
    LongHorizonRetentionPolicy,
    assess_long_horizon_retention,
)
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol
from nolane_personal.promotion_ceremony import (
    finalize_promotion_ceremony,
    verify_promotion_ceremony_against_registry,
)
from nolane_personal.serving_coordination import (
    ServingCoordinator,
    ServingLeasePolicy,
)
from nolane_personal.state import utc_now_iso
from nolane_personal.transactional_registry import (
    CheckpointRegistry,
    sha256_file,
)
from nolane_personal.store import payload_digest
from nolane_personal.unified_continual import (
    assess_unified_continual_chain,
    model_state_sha256,
)
from nolane_personal.unified_promotion_authority import (
    UnifiedPromotionAuthorizationPolicy,
    create_unified_operator_promotion_request,
    decide_unified_promotion_authorization,
    verify_unified_promotion_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixed_protocol():
    examples = [
        PersonalizationExample(
            prompt=f"prompt {i}",
            target=f"target {i}",
            language="vi" if i % 2 == 0 else "en",
        )
        for i in range(8)
    ]
    return build_personalization_protocol(
        examples,
        dataset_sha256=digest("fixed-dataset"),
        source_group_sha256=[
            digest(x) for x in ("a", "a", "b", "c", "d", "e", "f", "g")
        ],
    )


def l38_cycle(
    label: str,
    *,
    parent_sha: str,
    artifact_sha: str,
    boundary: str,
    cortex_before: str,
    cortex_after: str,
):
    before = model_state_sha256(
        boundary_state_digest=boundary,
        cortex_state_digest=cortex_before,
    )
    after = model_state_sha256(
        boundary_state_digest=boundary,
        cortex_state_digest=cortex_after,
    )
    court = assess_continual_learning(
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
    lineage = {
        "schema": "NOLANE-L38-CORTEX-UPDATE-LINEAGE-V1",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "retention_dataset_sha256": digest(f"{label}-retention-dataset"),
        "retention_protocol_sha256": digest(f"{label}-retention-protocol"),
        "retention_quality_court_sha256": digest(f"{label}-retention-quality"),
        "adaptation_dataset_sha256": digest(f"{label}-adaptation-dataset"),
        "adaptation_protocol_sha256": digest(f"{label}-adaptation-protocol"),
        "adaptation_quality_court_sha256": digest(f"{label}-adaptation-quality"),
    }
    lineage["lineage_sha256"] = payload_digest(lineage)
    core = {
        "schema": "NOLANE-L38-RECURRENT-CORTEX-CONTINUAL-UPDATE-V1",
        "authority": "RECURRENT_CORTEX_UPDATE_CANDIDATE_ONLY",
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
        "candidate_model_state_sha256_before": before,
        "candidate_model_state_sha256_after": after,
        "candidate_boundary_digest_before": boundary,
        "candidate_boundary_digest_after": boundary,
        "candidate_boundary_unchanged": True,
        "reference_boundary_digest_before": boundary,
        "reference_boundary_digest_after": boundary,
        "reference_boundary_unchanged": True,
        "candidate_cortex_digest_before": cortex_before,
        "candidate_cortex_digest_after": cortex_after,
        "candidate_cortex_changed": True,
        "reference_cortex_digest_before": cortex_before,
        "reference_cortex_digest_after": cortex_before,
        "reference_cortex_unchanged": True,
        "candidate_boundary_gradients_seen": 0,
        "candidate_cortex_gradients_seen": 4,
        "config": {},
        "continual_learning": court,
    }
    core["update_sha256"] = payload_digest(core)
    training = dict(core)
    training["lineage"] = lineage
    artifact = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": artifact_sha,
        "boundary_state_digest": boundary,
        "cortex_state_digest": cortex_after,
        "dataset_fingerprint": lineage["lineage_sha256"],
        "training_receipt": training,
    }
    return {
        "schema": "NOLANE-L38-RECURRENT-CORTEX-UPDATE-RUN-V1",
        "authority": "RECURRENT_CORTEX_UPDATE_EVIDENCE_ONLY_UNPROMOTED",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "training": training,
        "artifact": artifact,
    }


def evidence():
    parent = digest("artifact-0")
    artifact1 = digest("artifact-1")
    artifact2 = digest("artifact-2")
    boundary = digest("boundary")
    cortex0 = digest("cortex-0")
    cortex1 = digest("cortex-1")
    cortex2 = digest("cortex-2")
    cycles = [
        l38_cycle(
            "one",
            parent_sha=parent,
            artifact_sha=artifact1,
            boundary=boundary,
            cortex_before=cortex0,
            cortex_after=cortex1,
        ),
        l38_cycle(
            "two",
            parent_sha=artifact1,
            artifact_sha=artifact2,
            boundary=boundary,
            cortex_before=cortex1,
            cortex_after=cortex2,
        ),
    ]
    protocol = fixed_protocol()
    rows = protocol["splits"]["test"]
    initial = [1.0 + 0.1 * i for i in range(len(rows))]
    final = [x + 0.005 for x in initial]
    horizon = assess_long_horizon_retention(
        protocol,
        initial_checkpoint_sha256=parent,
        final_checkpoint_sha256=artifact2,
        initial_values=initial,
        final_values=final,
        policy=LongHorizonRetentionPolicy(
            max_overall_regression=0.01,
            max_worst_group_regression=0.02,
        ),
    )
    chain = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    return cycles, horizon, chain


def request(chain, horizon, *, approved=True, candidate=None):
    return create_unified_operator_promotion_request(
        active_parent_checkpoint_sha256=chain[
            "first_parent_checkpoint_sha256"
        ],
        candidate_checkpoint_sha256=(
            candidate or chain["final_artifact_checkpoint_sha256"]
        ),
        unified_chain_sha256=chain["chain_sha256"],
        long_horizon_retention_court_sha256=horizon["court_sha256"],
        approved=approved,
        nonce="0123456789abcdef-promote",
        created_at="2026-10-02T12:00:00+00:00",
    )


def test_unified_authorization_recomputes_and_authorizes_real_chain():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(chain, horizon),
        now="2026-10-02T12:05:00+00:00",
    )
    assert auth["status"] == "AUTHORIZED"
    assert auth["cycles"] == 2
    assert auth["cortex_cycles"] == 2
    assert auth["candidate_checkpoint_sha256"] == (
        chain["final_artifact_checkpoint_sha256"]
    )
    verify_unified_promotion_authorization(
        auth,
        now="2026-10-02T12:10:00+00:00",
    )


def test_unified_authorization_blocks_operator_denial():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(
            chain,
            horizon,
            approved=False,
        ),
        now="2026-10-02T12:05:00+00:00",
    )
    assert auth["status"] == "BLOCKED"
    assert "operator_did_not_approve" in auth["reasons"]


def test_unified_authorization_blocks_candidate_mismatch():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(
            chain,
            horizon,
            candidate=digest("wrong-candidate"),
        ),
        now="2026-10-02T12:05:00+00:00",
    )
    assert auth["status"] == "BLOCKED"
    assert "operator_candidate_checkpoint_mismatch" in auth["reasons"]


def test_unified_authorization_rejects_cycle_evidence_drift():
    cycles, horizon, chain = evidence()
    cycles[1]["training"]["optimizer_steps"] = 999
    with pytest.raises(ValueError):
        decide_unified_promotion_authorization(
            cycles=cycles,
            long_horizon_retention=horizon,
            unified_chain=chain,
            operator_request=request(chain, horizon),
            now="2026-10-02T12:05:00+00:00",
        )


def test_unified_authorization_enforces_cortex_evidence_policy():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(chain, horizon),
        policy=UnifiedPromotionAuthorizationPolicy(
            min_cycles=2,
            min_cortex_cycles=3,
        ),
        now="2026-10-02T12:05:00+00:00",
    )
    assert auth["status"] == "BLOCKED"
    assert "insufficient_recurrent_cortex_evidence" in auth["reasons"]


def test_unified_authorization_expiry_is_enforced():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(chain, horizon),
        policy=UnifiedPromotionAuthorizationPolicy(
            authorization_ttl_seconds=60,
        ),
        now="2026-10-02T12:05:00+00:00",
    )
    with pytest.raises(ValueError, match="expired"):
        verify_unified_promotion_authorization(
            auth,
            now="2026-10-02T12:06:01+00:00",
        )


def test_unified_authorization_tamper_is_detected():
    cycles, horizon, chain = evidence()
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=request(chain, horizon),
        now="2026-10-02T12:05:00+00:00",
    )
    auth["cortex_cycles"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_unified_promotion_authorization(auth)


def test_unified_request_never_persists_raw_nonce():
    _, horizon, chain = evidence()
    raw = "very-secret-operator-nonce-123"
    req = create_unified_operator_promotion_request(
        active_parent_checkpoint_sha256=chain[
            "first_parent_checkpoint_sha256"
        ],
        candidate_checkpoint_sha256=chain[
            "final_artifact_checkpoint_sha256"
        ],
        unified_chain_sha256=chain["chain_sha256"],
        long_horizon_retention_court_sha256=horizon["court_sha256"],
        approved=True,
        nonce=raw,
        created_at="2026-10-02T12:00:00+00:00",
    )
    assert raw not in str(req)



def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def integration_evidence(tmp_path):
    parent_bundle = tmp_path / "l40-parent-bundle"
    parent_bundle.mkdir()
    parent_checkpoint = parent_bundle / "factorized-nolane.pt"
    parent_checkpoint.write_bytes(b"l40-production-parent")
    parent_sha = sha256_file(parent_checkpoint)

    boundary = digest("l40-boundary")
    cortex0 = digest("l40-cortex-0")
    cortex1 = digest("l40-cortex-1")
    cortex2 = digest("l40-cortex-2")
    parent_manifest = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": parent_sha,
        "boundary_state_digest": boundary,
        "cortex_state_digest": cortex0,
        "dataset_fingerprint": digest("l40-parent-dataset"),
        "training_receipt": None,
    }
    write_json(
        parent_bundle / "factorized-nolane-manifest.json",
        parent_manifest,
    )

    final_bundle = tmp_path / "l40-final-bundle"
    final_bundle.mkdir()
    final_checkpoint = final_bundle / "factorized-nolane.pt"
    final_checkpoint.write_bytes(b"l40-final-l38-candidate")
    final_sha = sha256_file(final_checkpoint)

    artifact1 = digest("l40-offline-artifact-1")
    cycles = [
        l38_cycle(
            "integration-one",
            parent_sha=parent_sha,
            artifact_sha=artifact1,
            boundary=boundary,
            cortex_before=cortex0,
            cortex_after=cortex1,
        ),
        l38_cycle(
            "integration-two",
            parent_sha=artifact1,
            artifact_sha=final_sha,
            boundary=boundary,
            cortex_before=cortex1,
            cortex_after=cortex2,
        ),
    ]
    write_json(
        final_bundle / "factorized-nolane-manifest.json",
        cycles[-1]["artifact"],
    )
    write_json(
        final_bundle / "l38-run-receipt.json",
        cycles[-1],
    )

    frozen = fixed_protocol()
    rows = frozen["splits"]["test"]
    base = [1.0 + 0.1 * i for i in range(len(rows))]
    horizon = assess_long_horizon_retention(
        frozen,
        initial_checkpoint_sha256=parent_sha,
        final_checkpoint_sha256=final_sha,
        initial_values=base,
        final_values=[value + 0.005 for value in base],
        policy=LongHorizonRetentionPolicy(
            max_overall_regression=0.01,
            max_worst_group_regression=0.02,
        ),
    )
    chain = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    req = create_unified_operator_promotion_request(
        active_parent_checkpoint_sha256=parent_sha,
        candidate_checkpoint_sha256=final_sha,
        unified_chain_sha256=chain["chain_sha256"],
        long_horizon_retention_court_sha256=horizon["court_sha256"],
        approved=True,
        nonce="integration-nonce-0123456789",
        created_at=utc_now_iso(),
    )
    auth = decide_unified_promotion_authorization(
        cycles=cycles,
        long_horizon_retention=horizon,
        unified_chain=chain,
        operator_request=req,
        now=utc_now_iso(),
    )
    assert auth["status"] == "AUTHORIZED"
    return {
        "parent_bundle": parent_bundle,
        "parent_sha": parent_sha,
        "final_bundle": final_bundle,
        "final_sha": final_sha,
        "cycles": cycles,
        "horizon": horizon,
        "chain": chain,
        "authorization": auth,
    }


def test_l41_l40_authorized_l38_candidate_completes_release_ceremony(
    tmp_path,
):
    ev = integration_evidence(tmp_path)
    registry = CheckpointRegistry(tmp_path / "l41-registry")
    registry.initialize(ev["parent_bundle"])

    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        ev["authorization"],
        now=utc_now_iso(),
    )
    registry.verify_update(tx, now=utc_now_iso())
    pointer = registry.commit_update(tx, now=utc_now_iso())
    assert pointer["checkpoint_sha256"] == ev["final_sha"]

    coordinator = ServingCoordinator(
        registry,
        root=tmp_path / "l41-serving",
        policy=ServingLeasePolicy(
            lease_seconds=30.0,
            min_live_processes=1,
        ),
    )
    coordinator.register_loaded(
        "worker-l41",
        pointer=pointer,
        now=utc_now_iso(),
    )
    convergence = coordinator.assess_convergence(now=utc_now_iso())
    assert convergence["status"] == "PASS"

    ceremony = finalize_promotion_ceremony(
        registry,
        transaction_id=tx,
        convergence_receipt=convergence,
        now=utc_now_iso(),
    )
    assert ceremony["status"] == "COMPLETE"
    assert ceremony["authorization_schema"] == (
        "NOLANE-L40-UNIFIED-PROMOTION-AUTHORIZATION-V1"
    )
    assert ceremony["authorization_kind"] == "L40_UNIFIED_MODEL_CHAIN"
    assert ceremony["multicycle_chain_sha256"] == ev["chain"]["chain_sha256"]
    assert ceremony["candidate_checkpoint_sha256"] == ev["final_sha"]
    verify_promotion_ceremony_against_registry(registry, ceremony)

    audit = registry.verify_registry()
    assert audit["status"] == "PASS"
    assert audit["promotion_authorizations"] == 1
    assert audit["promotion_ceremonies"] == 1


def test_l41_l40_begin_rejects_final_model_state_mismatch(tmp_path):
    ev = integration_evidence(tmp_path)
    bad = dict(ev["authorization"])
    bad["final_model_state_sha256"] = digest("wrong-final-model-state")
    bad.pop("authorization_sha256")
    bad["authorization_sha256"] = payload_digest(bad)

    registry = CheckpointRegistry(tmp_path / "l41-state-mismatch")
    registry.initialize(ev["parent_bundle"])
    with pytest.raises(ValueError, match="final model-state mismatch"):
        registry.begin_authorized_update(
            ev["final_bundle"],
            bad,
            now=utc_now_iso(),
        )


def test_l41_l38_staging_receipt_tamper_fails_before_commit(tmp_path):
    ev = integration_evidence(tmp_path)
    registry = CheckpointRegistry(tmp_path / "l41-tamper")
    registry.initialize(ev["parent_bundle"])
    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        ev["authorization"],
        now=utc_now_iso(),
    )
    staged = registry._tx_dir(tx) / "staging" / "l38-run-receipt.json"
    payload = json.loads(staged.read_text(encoding="utf-8"))
    payload["training"]["optimizer_steps"] = 999
    write_json(staged, payload)

    with pytest.raises(ValueError):
        registry.verify_update(tx, now=utc_now_iso())
    assert registry.active_pointer()["checkpoint_sha256"] == ev["parent_sha"]



def run_l41_crash_probe(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def prepared_l41_transaction(tmp_path, label):
    ev = integration_evidence(tmp_path)
    registry = CheckpointRegistry(tmp_path / f"{label}-registry")
    parent = registry.initialize(ev["parent_bundle"])
    tx = registry.begin_authorized_update(
        ev["final_bundle"],
        ev["authorization"],
        now=utc_now_iso(),
    )
    registry.verify_update(tx, now=utc_now_iso())
    return ev, registry, parent, tx


def test_l41_hard_kill_l40_after_pointer_snapshot_recovers_abort(tmp_path):
    ev, registry, parent, tx = prepared_l41_transaction(
        tmp_path,
        "l40-snapshot-crash",
    )
    completed = run_l41_crash_probe(
        "scripts/crash_probe_checkpoint_commit.py",
        "--registry",
        str(registry.root),
        "--transaction",
        tx,
        "--crash-at",
        "POINTER_SNAPSHOT_WRITTEN",
        "--exit-code",
        "93",
    )
    assert completed.returncode == 93
    assert registry.lock_path.exists()
    assert registry.active_pointer() == parent

    recovered = registry.recover(break_stale_lock=True)
    assert len(recovered) == 1
    assert recovered[0]["state"] == "RECOVERED_ABORTED"
    assert registry.active_pointer()["checkpoint_sha256"] == ev["parent_sha"]
    assert registry.verify_registry()["status"] == "PASS"


def test_l41_hard_kill_l40_after_active_swap_recovers_commit(tmp_path):
    ev, registry, parent, tx = prepared_l41_transaction(
        tmp_path,
        "l40-active-swap-crash",
    )
    completed = run_l41_crash_probe(
        "scripts/crash_probe_checkpoint_commit.py",
        "--registry",
        str(registry.root),
        "--transaction",
        tx,
        "--crash-at",
        "ACTIVE_POINTER_SWAPPED",
        "--exit-code",
        "93",
    )
    assert completed.returncode == 93
    assert registry.lock_path.exists()
    active = registry.active_pointer()
    assert active["checkpoint_sha256"] == ev["final_sha"]
    assert active["pointer_sha256"] != parent["pointer_sha256"]

    recovered = registry.recover(break_stale_lock=True)
    assert len(recovered) == 1
    assert recovered[0]["state"] == "RECOVERED_COMMITTED"
    assert registry.active_pointer()["checkpoint_sha256"] == ev["final_sha"]
    assert registry.verify_registry()["status"] == "PASS"
