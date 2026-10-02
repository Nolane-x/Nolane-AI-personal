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
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol
from nolane_personal.store import payload_digest
from nolane_personal.unified_continual import (
    UnifiedContinualPolicy,
    assess_unified_continual_chain,
    model_state_sha256,
    normalize_continual_cycle,
    verify_l38_run_receipt,
    verify_unified_continual_chain_digest,
)


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


def continual(
    label: str,
    *,
    pre_state: str,
    post_state: str,
):
    receipt = assess_continual_learning(
        pre_update_checkpoint_sha256=pre_state,
        post_update_checkpoint_sha256=post_state,
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


def l31_cycle(
    label: str,
    *,
    parent_sha: str,
    artifact_sha: str,
    boundary_before: str,
    boundary_after: str,
    cortex_digest: str,
    adaptation_protocol_sha: str | None = None,
):
    l30 = continual(
        label,
        pre_state=boundary_before,
        post_state=boundary_after,
    )
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
        "continual_learning": l30,
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


def l38_cycle(
    label: str,
    *,
    parent_sha: str,
    artifact_sha: str,
    boundary_digest: str,
    cortex_before: str,
    cortex_after: str,
    adaptation_protocol_sha: str | None = None,
):
    model_before = model_state_sha256(
        boundary_state_digest=boundary_digest,
        cortex_state_digest=cortex_before,
    )
    model_after = model_state_sha256(
        boundary_state_digest=boundary_digest,
        cortex_state_digest=cortex_after,
    )
    l30 = continual(
        label,
        pre_state=model_before,
        post_state=model_after,
    )
    lineage = {
        "schema": "NOLANE-L38-CORTEX-UPDATE-LINEAGE-V1",
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
        "candidate_model_state_sha256_before": model_before,
        "candidate_model_state_sha256_after": model_after,
        "candidate_boundary_digest_before": boundary_digest,
        "candidate_boundary_digest_after": boundary_digest,
        "candidate_boundary_unchanged": True,
        "reference_boundary_digest_before": boundary_digest,
        "reference_boundary_digest_after": boundary_digest,
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
        "continual_learning": l30,
    }
    core["update_sha256"] = payload_digest(core)
    training = dict(core)
    training["lineage"] = lineage

    artifact = {
        "schema": "NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1",
        "authority": "FACTORIZED_CANDIDATE_UNPROMOTED",
        "checkpoint_sha256": artifact_sha,
        "boundary_state_digest": boundary_digest,
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


def horizon(initial_sha: str, final_sha: str, *, regress=0.005):
    frozen = protocol()
    rows = frozen["splits"]["test"]
    initial = [1.0 + index * 0.1 for index in range(len(rows))]
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


def mixed_chain():
    parent = digest("artifact-0")
    artifact1 = digest("artifact-1")
    artifact2 = digest("artifact-2")
    boundary0 = digest("boundary-0")
    boundary1 = digest("boundary-1")
    cortex0 = digest("cortex-0")
    cortex1 = digest("cortex-1")

    first = l31_cycle(
        "boundary-cycle",
        parent_sha=parent,
        artifact_sha=artifact1,
        boundary_before=boundary0,
        boundary_after=boundary1,
        cortex_digest=cortex0,
    )
    second = l38_cycle(
        "cortex-cycle",
        parent_sha=artifact1,
        artifact_sha=artifact2,
        boundary_digest=boundary1,
        cortex_before=cortex0,
        cortex_after=cortex1,
    )
    return [first, second], horizon(parent, artifact2)


def test_unified_chain_passes_exact_l31_to_l38_state_continuity():
    cycles, fixed = mixed_chain()
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    assert receipt["status"] == "PASS"
    assert receipt["cycles"] == 2
    assert receipt["boundary_cycles"] == 1
    assert receipt["cortex_cycles"] == 1
    assert (
        receipt["cycle_summaries"][0]["model_state_after"]
        == receipt["cycle_summaries"][1]["model_state_before"]
    )
    verify_unified_continual_chain_digest(receipt)
    verify_l38_run_receipt(cycles[1])


def test_unified_chain_blocks_cortex_state_discontinuity():
    cycles, fixed = mixed_chain()
    first = cycles[0]
    second = cycles[1]
    cycles[1] = l38_cycle(
        "cortex-cycle",
        parent_sha=first["artifact"]["checkpoint_sha256"],
        artifact_sha=second["artifact"]["checkpoint_sha256"],
        boundary_digest=first["training"]["candidate_boundary_digest_after"],
        cortex_before=digest("unrelated-cortex"),
        cortex_after=digest("cortex-2"),
    )
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    assert receipt["status"] == "BLOCKED"
    assert "cortex_state_chain_discontinuity" in receipt["reasons"]
    assert "model_state_chain_discontinuity" in receipt["reasons"]


def test_unified_chain_requires_at_least_one_cortex_cycle():
    parent = digest("artifact-0")
    artifact1 = digest("artifact-1")
    artifact2 = digest("artifact-2")
    cortex = digest("stable-cortex")
    first = l31_cycle(
        "one",
        parent_sha=parent,
        artifact_sha=artifact1,
        boundary_before=digest("boundary-0"),
        boundary_after=digest("boundary-1"),
        cortex_digest=cortex,
    )
    second = l31_cycle(
        "two",
        parent_sha=artifact1,
        artifact_sha=artifact2,
        boundary_before=digest("boundary-1"),
        boundary_after=digest("boundary-2"),
        cortex_digest=cortex,
    )
    receipt = assess_unified_continual_chain(
        [first, second],
        long_horizon_retention=horizon(parent, artifact2),
    )
    assert receipt["status"] == "BLOCKED"
    assert "insufficient_recurrent_cortex_cycles" in receipt["reasons"]


def test_unified_chain_blocks_adaptation_protocol_replay_across_update_types():
    cycles, fixed = mixed_chain()
    repeated = cycles[0]["training"]["lineage"]["adaptation_protocol_sha256"]
    cycles[1] = l38_cycle(
        "cortex-cycle",
        parent_sha=cycles[0]["artifact"]["checkpoint_sha256"],
        artifact_sha=cycles[1]["artifact"]["checkpoint_sha256"],
        boundary_digest=cycles[0]["training"]["candidate_boundary_digest_after"],
        cortex_before=cycles[0]["training"]["candidate_cortex_digest_after"],
        cortex_after=digest("cortex-1"),
        adaptation_protocol_sha=repeated,
    )
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    assert receipt["status"] == "BLOCKED"
    assert "adaptation_protocol_reused_across_cycles" in receipt["reasons"]


def test_unified_chain_blocks_artifact_discontinuity():
    cycles, fixed = mixed_chain()
    cycles[1]["parent_factorized_checkpoint_sha256"] = digest("wrong-parent")
    cycles[1]["training"]["lineage"]["parent_factorized_checkpoint_sha256"] = (
        digest("wrong-parent")
    )
    lineage = cycles[1]["training"]["lineage"]
    lineage.pop("lineage_sha256")
    lineage["lineage_sha256"] = payload_digest(lineage)
    cycles[1]["artifact"]["dataset_fingerprint"] = lineage["lineage_sha256"]
    cycles[1]["artifact"]["training_receipt"] = cycles[1]["training"]

    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    assert receipt["status"] == "BLOCKED"
    assert "artifact_chain_discontinuity" in receipt["reasons"]


def test_l38_verifier_rejects_model_state_tamper():
    cycles, _ = mixed_chain()
    cycle = cycles[1]
    cycle["training"] = dict(cycle["training"])
    cycle["training"]["candidate_model_state_sha256_after"] = digest("tampered")
    cycle["artifact"] = dict(cycle["artifact"])
    cycle["artifact"]["training_receipt"] = cycle["training"]

    with pytest.raises(ValueError):
        verify_l38_run_receipt(cycle)


def test_unified_chain_blocks_failed_fixed_panel():
    cycles, _ = mixed_chain()
    fixed = horizon(
        cycles[0]["parent_factorized_checkpoint_sha256"],
        cycles[-1]["artifact"]["checkpoint_sha256"],
        regress=0.10,
    )
    assert fixed["status"] == "BLOCKED"
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    assert receipt["status"] == "BLOCKED"
    assert "long_horizon_fixed_panel_failed" in receipt["reasons"]


def test_unified_chain_digest_tamper_is_detected():
    cycles, fixed = mixed_chain()
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
    )
    receipt["cortex_cycles"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_unified_continual_chain_digest(receipt)


def test_policy_can_require_multiple_cortex_cycles():
    cycles, fixed = mixed_chain()
    receipt = assess_unified_continual_chain(
        cycles,
        long_horizon_retention=fixed,
        policy=UnifiedContinualPolicy(
            min_cycles=2,
            min_cortex_cycles=2,
        ),
    )
    assert receipt["status"] == "BLOCKED"
    assert "insufficient_recurrent_cortex_cycles" in receipt["reasons"]


def test_normalization_identifies_update_ownership():
    cycles, _ = mixed_chain()
    assert normalize_continual_cycle(cycles[0])["cycle_type"] == "BOUNDARY_UPDATE"
    assert normalize_continual_cycle(cycles[1])["cycle_type"] == "CORTEX_UPDATE"
