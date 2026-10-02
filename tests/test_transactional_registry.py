import hashlib
import json

import pytest

from nolane_personal.continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
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


def initial_bundle(tmp_path, label="parent"):
    bundle = tmp_path / f"{label}-bundle"
    bundle.mkdir()
    checkpoint = bundle / "factorized-nolane.pt"
    checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
    sha = sha256_file(checkpoint)
    manifest = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": sha,
        "boundary_state_digest": digest(f"{label}-boundary"),
        "cortex_state_digest": digest("stable-cortex"),
        "dataset_fingerprint": digest(f"{label}-dataset"),
        "training_receipt": None,
    }
    write_json(bundle / "factorized-nolane-manifest.json", manifest)
    return bundle, sha, manifest


def candidate_bundle(
    tmp_path,
    *,
    parent_sha: str,
    label="candidate",
    boundary_before=None,
):
    bundle = tmp_path / f"{label}-bundle"
    bundle.mkdir()
    checkpoint = bundle / "factorized-nolane.pt"
    checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
    checkpoint_sha = sha256_file(checkpoint)

    boundary_before = boundary_before or digest(f"{label}-before")
    boundary_after = digest(f"{label}-after")
    cortex_digest = digest("stable-cortex")

    continual = assess_continual_learning(
        pre_update_checkpoint_sha256=boundary_before,
        post_update_checkpoint_sha256=boundary_after,
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
    assert continual["status"] == "PASS"

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
        "candidate_cortex_digest_before": cortex_digest,
        "candidate_cortex_digest_after": cortex_digest,
        "candidate_cortex_unchanged": True,
        "reference_cortex_digest_before": cortex_digest,
        "reference_cortex_digest_after": cortex_digest,
        "reference_cortex_unchanged": True,
        "candidate_boundary_gradients_seen": 4,
        "candidate_cortex_gradients_seen": 0,
        "config": {},
        "continual_learning": continual,
        "lineage": lineage,
    }
    manifest = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": checkpoint_sha,
        "boundary_state_digest": boundary_after,
        "cortex_state_digest": cortex_digest,
        "dataset_fingerprint": lineage["lineage_sha256"],
        "training_receipt": training,
    }
    run = {
        "schema": "NOLANE-L31-CONTINUAL-FACTORIZED-UPDATE-RUN-V1",
        "authority": "CONTINUAL_UPDATE_EVIDENCE_ONLY_UNPROMOTED",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "training": training,
        "artifact": manifest,
    }
    write_json(bundle / "factorized-nolane-manifest.json", manifest)
    write_json(bundle / "l31-run-receipt.json", run)
    return bundle, checkpoint_sha, boundary_after


def test_transactional_registry_commits_verified_l31_candidate(tmp_path):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    p0 = registry.initialize(parent)
    assert p0["generation"] == 0
    assert p0["checkpoint_sha256"] == parent_sha

    candidate, candidate_sha, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    verified = registry.verify_update(tx)
    assert verified["state"] == "VERIFIED"

    p1 = registry.commit_update(tx)
    assert p1["generation"] == 1
    assert p1["checkpoint_sha256"] == candidate_sha
    assert p1["previous_pointer_sha256"] == p0["pointer_sha256"]
    assert [event["state"] for event in registry.transaction_events(tx)] == [
        "PREPARED",
        "VERIFIED",
        "POINTER_SWAPPED",
        "COMMITTED",
    ]
    audit = registry.verify_registry()
    assert audit["status"] == "PASS"
    assert audit["active_generation"] == 1


def test_recovery_aborts_when_crash_happens_before_pointer_swap(tmp_path):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    p0 = registry.initialize(parent)
    candidate, _, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    registry.verify_update(tx)

    recovered = registry.recover()
    assert len(recovered) == 1
    assert recovered[0]["state"] == "RECOVERED_ABORTED"
    assert registry.active_pointer() == p0
    assert registry.transaction_events(tx)[-1]["state"] == "RECOVERED_ABORTED"
    assert registry.verify_registry()["status"] == "PASS"


def test_recovery_finishes_commit_when_crash_occurs_after_atomic_swap(
    tmp_path,
    monkeypatch,
):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    candidate, candidate_sha, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    registry.verify_update(tx)

    original = registry._append_event

    def crash_after_swap(transaction_id, *, state, details):
        if state == "POINTER_SWAPPED":
            raise RuntimeError("simulated process death after pointer swap")
        return original(
            transaction_id,
            state=state,
            details=details,
        )

    monkeypatch.setattr(registry, "_append_event", crash_after_swap)
    with pytest.raises(RuntimeError, match="simulated process death"):
        registry.commit_update(tx)

    active = registry.active_pointer()
    assert active["checkpoint_sha256"] == candidate_sha
    assert active["transaction_id"] == tx

    monkeypatch.setattr(registry, "_append_event", original)
    recovered = registry.recover()
    assert len(recovered) == 1
    assert recovered[0]["state"] == "RECOVERED_COMMITTED"
    assert registry.active_pointer()["checkpoint_sha256"] == candidate_sha
    assert registry.verify_registry()["status"] == "PASS"


def test_staged_candidate_tamper_is_detected_before_commit(tmp_path):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    candidate, _, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    staged = registry._tx_dir(tx) / "staging" / "l31-run-receipt.json"
    payload = json.loads(staged.read_text(encoding="utf-8"))
    payload["training"]["optimizer_steps"] = 999
    write_json(staged, payload)

    with pytest.raises(ValueError):
        registry.verify_update(tx)
    assert registry.active_pointer()["generation"] == 0


def test_rollback_is_forward_auditable_pointer_transition(tmp_path):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    p0 = registry.initialize(parent)
    candidate, candidate_sha, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    registry.verify_update(tx)
    p1 = registry.commit_update(tx)
    assert p1["checkpoint_sha256"] == candidate_sha

    receipt = registry.rollback_to_generation(0)
    p2 = registry.active_pointer()
    assert receipt["from_generation"] == 1
    assert receipt["target_generation"] == 0
    assert receipt["new_generation"] == 2
    assert p2["generation"] == 2
    assert p2["checkpoint_sha256"] == parent_sha
    assert p2["transition"] == "ROLLBACK"
    assert p2["rollback_target_generation"] == 0
    assert p2["previous_pointer_sha256"] == p1["pointer_sha256"]
    assert p2["pointer_sha256"] != p0["pointer_sha256"]
    assert registry.verify_registry()["status"] == "PASS"


def test_registry_lock_requires_explicit_stale_recovery(tmp_path):
    parent, _, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    registry.lock_path.write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="registry is locked"):
        registry.recover()

    assert registry.recover(break_stale_lock=True) == []
    assert not registry.lock_path.exists()


def test_registry_audit_detects_active_pointer_tamper(tmp_path):
    parent, _, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    active = json.loads(registry.active_path.read_text(encoding="utf-8"))
    active["generation"] = 9
    write_json(registry.active_path, active)

    with pytest.raises(ValueError, match="pointer digest mismatch"):
        registry.verify_registry()


def test_begin_rejects_candidate_from_nonactive_parent(tmp_path):
    parent, _, _ = initial_bundle(tmp_path, label="parent")
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    candidate, _, _ = candidate_bundle(
        tmp_path,
        parent_sha=digest("different-parent"),
        label="wrong-parent-candidate",
    )

    with pytest.raises(ValueError, match="parent does not match active checkpoint"):
        registry.begin_l31_update(candidate)


def test_registry_audit_detects_rollback_receipt_tamper(tmp_path):
    parent, parent_sha, _ = initial_bundle(tmp_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.initialize(parent)
    candidate, _, _ = candidate_bundle(
        tmp_path,
        parent_sha=parent_sha,
    )
    tx = registry.begin_l31_update(candidate)
    registry.verify_update(tx)
    registry.commit_update(tx)
    receipt = registry.rollback_to_generation(0)

    path = registry.rollbacks_dir / f"{int(receipt['new_generation']):012d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_generation"] = 999
    write_json(path, payload)

    with pytest.raises(ValueError, match="rollback receipt digest mismatch"):
        registry.verify_registry()
