import hashlib
import json
import math

import pytest

from nolane_personal.continual_learning_court import (
    ContinualLearningPolicy,
    assess_continual_learning,
    effective_continual_promotion_status,
    verify_continual_learning_digest,
    verify_continual_learning_receipt,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence():
    return {
        "pre": digest("checkpoint-before"),
        "post": digest("checkpoint-after"),
        "retention_groups": [
            digest("old-a"),
            digest("old-a"),
            digest("old-b"),
            digest("old-c"),
        ],
        "retention_before": [1.00, 1.10, 0.90, 1.20],
        "retention_after": [1.01, 1.11, 0.91, 1.21],
        "adaptation_groups": [
            digest("new-a"),
            digest("new-b"),
            digest("new-c"),
        ],
        "adaptation_before": [1.30, 1.20, 1.40],
        "adaptation_after": [1.10, 1.05, 1.15],
    }


def assess(data=None, policy=None):
    data = data or evidence()
    return assess_continual_learning(
        pre_update_checkpoint_sha256=data["pre"],
        post_update_checkpoint_sha256=data["post"],
        retention_group_sha256=data["retention_groups"],
        retention_before_values=data["retention_before"],
        retention_after_values=data["retention_after"],
        adaptation_group_sha256=data["adaptation_groups"],
        adaptation_before_values=data["adaptation_before"],
        adaptation_after_values=data["adaptation_after"],
        policy=policy,
    )


def test_continual_learning_passes_stable_old_groups_and_new_gain():
    data = evidence()
    receipt = assess(
        data,
        ContinualLearningPolicy(
            max_worst_retention_regression=0.02,
            max_mean_retention_regression=0.02,
            min_mean_adaptation_gain=0.10,
            max_worst_adaptation_regression=0.0,
        ),
    )
    assert receipt["status"] == "PASS"
    assert receipt["retention"]["groups"] == 3
    assert receipt["adaptation"]["groups"] == 3
    assert (
        receipt["retention"]["summary"]["worst_group_regression"]
        <= 0.02
    )
    assert receipt["adaptation"]["summary"]["mean_group_gain"] >= 0.10

    rendered = json.dumps(receipt, sort_keys=True)
    for raw in set(
        data["retention_groups"] + data["adaptation_groups"]
    ):
        assert raw not in rendered

    verify_continual_learning_receipt(
        receipt,
        pre_update_checkpoint_sha256=data["pre"],
        post_update_checkpoint_sha256=data["post"],
        retention_group_sha256=data["retention_groups"],
        retention_before_values=data["retention_before"],
        retention_after_values=data["retention_after"],
        adaptation_group_sha256=data["adaptation_groups"],
        adaptation_before_values=data["adaptation_before"],
        adaptation_after_values=data["adaptation_after"],
    )


def test_continual_learning_blocks_average_hiding_one_forgotten_group():
    data = evidence()
    data["retention_before"] = [1.0, 1.0, 1.0, 1.0]
    # One group improves strongly, one is stable, one regresses badly.
    # Overall example regression is still close to zero.
    data["retention_after"] = [0.90, 0.90, 1.00, 1.18]
    receipt = assess(
        data,
        ContinualLearningPolicy(
            max_worst_retention_regression=0.03,
            max_mean_retention_regression=0.10,
            min_mean_adaptation_gain=0.0,
            max_worst_adaptation_regression=0.03,
        ),
    )
    assert receipt["retention"]["summary"]["overall_regression"] == pytest.approx(
        -0.005
    )
    assert receipt["retention"]["summary"]["worst_group_regression"] == pytest.approx(
        0.18
    )
    assert receipt["status"] == "BLOCKED"
    assert "worst_retention_forgetting_failed" in receipt["reasons"]


def test_continual_learning_blocks_new_group_regression_hidden_by_average():
    data = evidence()
    data["adaptation_before"] = [1.0, 1.0, 1.0]
    data["adaptation_after"] = [0.70, 0.70, 1.20]
    receipt = assess(
        data,
        ContinualLearningPolicy(
            min_mean_adaptation_gain=0.05,
            max_worst_adaptation_regression=0.05,
        ),
    )
    assert receipt["adaptation"]["summary"]["mean_group_gain"] > 0.05
    assert receipt["adaptation"]["summary"]["worst_group_regression"] == pytest.approx(
        0.20
    )
    assert receipt["status"] == "BLOCKED"
    assert "worst_adaptation_group_failed" in receipt["reasons"]


def test_continual_learning_requires_disjoint_old_and_new_groups():
    data = evidence()
    data["adaptation_groups"][0] = data["retention_groups"][0]
    receipt = assess(data)
    assert receipt["status"] == "BLOCKED"
    assert "retention_adaptation_group_overlap" in receipt["reasons"]


def test_continual_learning_requires_a_real_checkpoint_change():
    data = evidence()
    data["post"] = data["pre"]
    receipt = assess(data)
    assert receipt["status"] == "BLOCKED"
    assert "checkpoint_did_not_change" in receipt["reasons"]


def test_continual_learning_rejects_length_and_nonfinite_metric_drift():
    data = evidence()
    with pytest.raises(ValueError, match="lengths do not match"):
        assess_continual_learning(
            pre_update_checkpoint_sha256=data["pre"],
            post_update_checkpoint_sha256=data["post"],
            retention_group_sha256=data["retention_groups"],
            retention_before_values=data["retention_before"][:-1],
            retention_after_values=data["retention_after"],
            adaptation_group_sha256=data["adaptation_groups"],
            adaptation_before_values=data["adaptation_before"],
            adaptation_after_values=data["adaptation_after"],
        )

    data = evidence()
    data["adaptation_after"][0] = math.inf
    with pytest.raises(ValueError, match="metrics must be finite"):
        assess(data)


def test_continual_learning_digest_tamper_is_detected():
    receipt = assess()
    verify_continual_learning_digest(receipt)
    receipt["retention"]["summary"]["worst_group_regression"] = 9.0
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_continual_learning_digest(receipt)


def test_effective_continual_promotion_status_fails_closed():
    receipt = assess()
    evaluation = {
        "decision": {"status": "CANDIDATE_UPDATE_QUALITY_PASS"},
        "continual_learning": receipt,
    }
    assert (
        effective_continual_promotion_status(evaluation)
        == "CANDIDATE_UPDATE_QUALITY_PASS"
    )

    missing = {"decision": {"status": "CANDIDATE_UPDATE_QUALITY_PASS"}}
    assert effective_continual_promotion_status(missing) == ""

    blocked = assess(
        policy=ContinualLearningPolicy(
            min_mean_adaptation_gain=10.0,
        )
    )
    evaluation["continual_learning"] = blocked
    assert effective_continual_promotion_status(evaluation) == ""

    tampered = assess()
    tampered["status"] = "PASS"
    tampered["retention"]["summary"]["worst_group_regression"] = 9.0
    evaluation["continual_learning"] = tampered
    assert effective_continual_promotion_status(evaluation) == ""
