import hashlib

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
    verify_l31_run_receipt,
    verify_multicycle_chain_digest,
)
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol
from nolane_personal.store import payload_digest


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def make_cycle(
    label: str,
    *,
    parent_sha: str,
    boundary_before: str,
    boundary_after: str,
    artifact_sha: str,
    adaptation_protocol_sha: str | None = None,
):
    retention_groups = [digest(f"{label}-old-a"), digest(f"{label}-old-b")]
    adaptation_groups = [digest(f"{label}-new-a"), digest(f"{label}-new-b")]
    continual = assess_continual_learning(
        pre_update_checkpoint_sha256=boundary_before,
        post_update_checkpoint_sha256=boundary_after,
        retention_group_sha256=retention_groups,
        retention_before_values=[1.0, 1.1],
        retention_after_values=[1.005, 1.105],
        adaptation_group_sha256=adaptation_groups,
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
        "adaptation_protocol_sha256": (
            adaptation_protocol_sha
            or digest(f"{label}-adaptation-protocol")
        ),
        "adaptation_quality_court_sha256": digest(f"{label}-adaptation-quality"),
    }
    lineage["lineage_sha256"] = payload_digest(lineage)

    cortex_digest = digest("stable-cortex")
    training = {
        "schema": "NOLANE-L31-FACTORIZED-CONTINUAL-UPDATE-V1",
        "authority": "CONTINUAL_FACTORIZED_UPDATE_CANDIDATE_ONLY",
        "adaptation_train_examples": 4,
        "retention_rehearsal_examples": 6,
        "retention_eval_examples": 2,
        "adaptation_eval_examples": 2,
        "epochs": 2,
        "optimizer_steps": 8,
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
        "candidate_boundary_gradients_seen": 8,
        "candidate_cortex_gradients_seen": 0,
        "config": {},
        "continual_learning": continual,
        "lineage": lineage,
    }
    artifact = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": artifact_sha,
        "boundary_state_digest": boundary_after,
        "cortex_state_digest": cortex_digest,
        "dataset_fingerprint": lineage["lineage_sha256"],
        "training_receipt": training,
    }
    return {
        "schema": "NOLANE-L31-CONTINUAL-FACTORIZED-UPDATE-RUN-V1",
        "authority": "CONTINUAL_UPDATE_EVIDENCE_ONLY_UNPROMOTED",
        "parent_factorized_checkpoint_sha256": parent_sha,
        "training": training,
        "artifact": artifact,
    }


def long_horizon(initial_sha: str, final_sha: str, *, regress=0.005):
    frozen = protocol()
    rows = frozen["splits"]["test"]
    initial = [1.0 + 0.1 * index for index in range(len(rows))]
    final = [value + regress for value in initial]
    return assess_long_horizon_retention(
        frozen,
        initial_checkpoint_sha256=initial_sha,
        final_checkpoint_sha256=final_sha,
        initial_values=initial,
        final_values=final,
        policy=LongHorizonRetentionPolicy(
            max_overall_regression=0.01,
            max_worst_group_regression=0.02,
        ),
    )


def chain():
    parent = digest("parent-artifact")
    b0 = digest("boundary-0")
    b1 = digest("boundary-1")
    b2 = digest("boundary-2")
    a1 = digest("artifact-1")
    a2 = digest("artifact-2")
    c1 = make_cycle(
        "cycle-1",
        parent_sha=parent,
        boundary_before=b0,
        boundary_after=b1,
        artifact_sha=a1,
    )
    c2 = make_cycle(
        "cycle-2",
        parent_sha=a1,
        boundary_before=b1,
        boundary_after=b2,
        artifact_sha=a2,
    )
    return [c1, c2], long_horizon(parent, a2)


def test_multicycle_chain_passes_exact_artifact_and_boundary_lineage():
    cycles, horizon = chain()
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "PASS"
    assert receipt["cycles"] == 2
    assert receipt["summary"]["unique_adaptation_protocols"] == 2
    assert receipt["summary"]["positive_mean_retention_regression_sum"] > 0
    verify_multicycle_chain_digest(receipt)
    for cycle in cycles:
        verify_l31_run_receipt(cycle)


def test_multicycle_chain_blocks_artifact_discontinuity():
    cycles, horizon = chain()
    wrong_parent = digest("wrong-parent")
    cycles[1] = make_cycle(
        "cycle-2",
        parent_sha=wrong_parent,
        boundary_before=cycles[0]["training"]["candidate_boundary_digest_after"],
        boundary_after=digest("boundary-2"),
        artifact_sha=cycles[1]["artifact"]["checkpoint_sha256"],
    )
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "BLOCKED"
    assert "artifact_chain_discontinuity" in receipt["reasons"]


def test_multicycle_chain_blocks_boundary_state_discontinuity():
    cycles, horizon = chain()
    cycles[1] = make_cycle(
        "cycle-2",
        parent_sha=cycles[0]["artifact"]["checkpoint_sha256"],
        boundary_before=digest("unrelated-boundary"),
        boundary_after=digest("boundary-2"),
        artifact_sha=cycles[1]["artifact"]["checkpoint_sha256"],
    )
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "BLOCKED"
    assert "boundary_state_chain_discontinuity" in receipt["reasons"]


def test_multicycle_chain_blocks_reused_adaptation_protocol():
    cycles, horizon = chain()
    repeated = cycles[0]["training"]["lineage"]["adaptation_protocol_sha256"]
    cycles[1] = make_cycle(
        "cycle-2",
        parent_sha=cycles[0]["artifact"]["checkpoint_sha256"],
        boundary_before=cycles[0]["training"]["candidate_boundary_digest_after"],
        boundary_after=digest("boundary-2"),
        artifact_sha=cycles[1]["artifact"]["checkpoint_sha256"],
        adaptation_protocol_sha=repeated,
    )
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "BLOCKED"
    assert "adaptation_protocol_reused_across_cycles" in receipt["reasons"]


def test_multicycle_chain_requires_fixed_panel_to_match_chain_endpoints():
    cycles, _ = chain()
    horizon = long_horizon(
        digest("wrong-initial"),
        cycles[-1]["artifact"]["checkpoint_sha256"],
    )
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "BLOCKED"
    assert "long_horizon_initial_checkpoint_mismatch" in receipt["reasons"]


def test_multicycle_chain_blocks_failed_fixed_panel():
    cycles, _ = chain()
    horizon = long_horizon(
        cycles[0]["parent_factorized_checkpoint_sha256"],
        cycles[-1]["artifact"]["checkpoint_sha256"],
        regress=0.10,
    )
    assert horizon["status"] == "BLOCKED"
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    assert receipt["status"] == "BLOCKED"
    assert "long_horizon_fixed_panel_failed" in receipt["reasons"]


def test_multicycle_chain_detects_tampering():
    cycles, horizon = chain()
    receipt = assess_multicycle_continual_chain(
        cycles,
        long_horizon_retention=horizon,
    )
    receipt["summary"]["worst_cycle_retention_group_regression"] = 9.0
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_multicycle_chain_digest(receipt)


def test_l31_run_verifier_rejects_artifact_training_receipt_drift():
    cycles, _ = chain()
    cycle = cycles[0]
    cycle["artifact"] = dict(cycle["artifact"])
    cycle["artifact"]["training_receipt"] = dict(cycle["training"])
    cycle["artifact"]["training_receipt"]["optimizer_steps"] = 999
    with pytest.raises(ValueError, match="artifact training receipt mismatch"):
        verify_l31_run_receipt(cycle)
