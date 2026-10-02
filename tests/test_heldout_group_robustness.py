import hashlib
import json

import pytest

from nolane_personal.heldout_group_robustness import (
    HeldoutGroupRobustnessPolicy,
    assess_group_robustness,
    verify_group_robustness_receipt,
)
from nolane_personal.personal_dataset import PersonalizationExample
from nolane_personal.personal_protocol import build_personalization_protocol


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protocol():
    examples=[
        PersonalizationExample(
            prompt=f"prompt {i}",
            target=f"target {i}",
            language="vi" if i%2==0 else "en",
        )
        for i in range(8)
    ]
    groups=[digest(name) for name in ("a","a","b","c","d","e","f","g")]
    return build_personalization_protocol(
        examples,
        dataset_sha256="dataset",
        source_group_sha256=groups,
    )


def test_group_robustness_passes_when_every_heldout_group_is_noninferior():
    frozen=protocol()
    rows=frozen["splits"]["test"]
    assert len(rows)>=2
    ref=[1.0+0.1*i for i in range(len(rows))]
    cand=[value+0.01 for value in ref]
    receipt=assess_group_robustness(
        frozen,
        split="test",
        reference_values=ref,
        candidate_values=cand,
        policy=HeldoutGroupRobustnessPolicy(max_worst_group_regression=0.02),
    )
    assert receipt["status"]=="PASS"
    assert receipt["groups"]>=2
    assert receipt["summary"]["worst_group_regression"]<=0.02
    rendered=json.dumps(receipt,sort_keys=True)
    for raw in ("prompt","target",digest("f"),digest("g")):
        assert raw not in rendered
    verify_group_robustness_receipt(
        receipt,
        protocol=frozen,
        reference_values=ref,
        candidate_values=cand,
    )


def test_group_robustness_blocks_average_that_hides_bad_group():
    frozen=protocol()
    rows=frozen["splits"]["test"]
    assert len(rows)==2
    ref=[1.0,1.0]
    # Global regression is only +0.01, but one held-out group regresses +0.10.
    cand=[0.92,1.10]
    receipt=assess_group_robustness(
        frozen,
        split="test",
        reference_values=ref,
        candidate_values=cand,
        policy=HeldoutGroupRobustnessPolicy(max_worst_group_regression=0.03),
    )
    assert receipt["summary"]["overall_regression"]==pytest.approx(0.01)
    assert receipt["summary"]["worst_group_regression"]==pytest.approx(0.10)
    assert receipt["status"]=="BLOCKED"
    assert "worst_group_noninferiority_failed" in receipt["reasons"]


def test_group_robustness_requires_two_independent_heldout_groups():
    examples=[
        PersonalizationExample(prompt=f"p{i}",target=f"t{i}",language="vi")
        for i in range(7)
    ]
    groups=[digest(name) for name in ("a","b","c","d","e","f","f")]
    frozen=build_personalization_protocol(
        examples,
        dataset_sha256="dataset",
        source_group_sha256=groups,
    )
    rows=frozen["splits"]["test"]
    assert len({row["source_group_sha256"] for row in rows})==1
    receipt=assess_group_robustness(
        frozen,
        split="test",
        reference_values=[1.0]*len(rows),
        candidate_values=[1.0]*len(rows),
    )
    assert receipt["status"]=="BLOCKED"
    assert "insufficient_heldout_source_groups" in receipt["reasons"]


def test_group_robustness_rejects_metric_length_drift():
    frozen=protocol()
    with pytest.raises(ValueError,match="metric length"):
        assess_group_robustness(
            frozen,
            split="test",
            reference_values=[1.0],
            candidate_values=[1.0],
        )
