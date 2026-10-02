import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from nolane_personal.continual_factorized_update import (
    ContinualFactorizedUpdateConfig,
    train_continual_factorized_update,
)
from nolane_personal.continual_learning_court import ContinualLearningPolicy
from nolane_personal.deep_recurrent_cortex import (
    DeepRecurrentCortexConfig,
    DeepRecurrentStateSpaceCortex,
)
from nolane_personal.factorized_boundary import factorize_standalone_boundary
from nolane_personal.standalone_model import (
    StandaloneBoundaryModule,
    StandaloneNolaneConfig,
    StandaloneNolaneLM,
)
from nolane_personal.surgery import module_parameter_digest


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_boundary(*, vocab=47, hidden=16):
    torch.manual_seed(17)
    return StandaloneBoundaryModule(
        StandaloneNolaneConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def model(boundary, seed=23):
    cortex = DeepRecurrentStateSpaceCortex(
        boundary.config.hidden_size,
        DeepRecurrentCortexConfig(
            state_dim=8,
            latent_dim=32,
            virtual_steps=3,
            max_virtual_steps=4,
        ),
        seed=seed,
    )
    return StandaloneNolaneLM(boundary, cortex, [0.2] * 32)


def cloned_pair():
    source = source_boundary()
    ref_boundary, _ = factorize_standalone_boundary(
        source,
        rank=5,
        seed=19,
    )
    candidate_boundary, _ = factorize_standalone_boundary(
        source,
        rank=5,
        seed=19,
    )
    reference = model(ref_boundary)
    candidate = model(candidate_boundary)
    candidate.boundary.module.load_state_dict(reference.boundary.module.state_dict())
    candidate.cortex.module.load_state_dict(reference.cortex.module.state_dict())
    return candidate, reference


def example(tokens):
    ids = list(tokens)
    return (
        ids,
        [-100] + [3] * (len(ids) - 1),
        [0.2] * 32,
        1.0,
    )


def permissive_policy():
    return ContinualLearningPolicy(
        min_retention_groups=2,
        min_adaptation_groups=2,
        max_worst_retention_regression=10.0,
        max_mean_retention_regression=10.0,
        min_mean_adaptation_gain=-10.0,
        max_worst_adaptation_regression=10.0,
    )


def test_continual_factorized_update_changes_only_candidate_boundary():
    candidate, reference = cloned_pair()
    before_reference_boundary = module_parameter_digest(reference.boundary.module)
    before_reference_cortex = module_parameter_digest(reference.cortex.module)
    before_candidate_cortex = module_parameter_digest(candidate.cortex.module)

    retention = [
        example([1, 5, 6, 7]),
        example([1, 8, 9, 10]),
    ]
    adaptation = [
        example([1, 11, 12, 13]),
        example([1, 14, 15, 16]),
    ]
    old_groups = [digest("old-a"), digest("old-b")]
    new_groups = [digest("new-a"), digest("new-b")]

    receipt = train_continual_factorized_update(
        candidate,
        reference,
        adaptation,
        retention,
        adaptation_group_sha256=new_groups,
        retention_group_sha256=old_groups,
        config=ContinualFactorizedUpdateConfig(
            epochs=2,
            learning_rate=0.01,
            retention_distill_weight=1.0,
        ),
        court_policy=permissive_policy(),
    )

    assert receipt.candidate_boundary_changed
    assert receipt.candidate_boundary_gradients_seen > 0
    assert receipt.candidate_cortex_unchanged
    assert receipt.candidate_cortex_gradients_seen == 0
    assert receipt.reference_boundary_unchanged
    assert receipt.reference_cortex_unchanged
    assert receipt.continual_learning["status"] == "PASS"

    assert module_parameter_digest(reference.boundary.module) == before_reference_boundary
    assert module_parameter_digest(reference.cortex.module) == before_reference_cortex
    assert module_parameter_digest(candidate.cortex.module) == before_candidate_cortex

    rendered = json.dumps(receipt.to_dict(), sort_keys=True)
    for raw in old_groups + new_groups:
        assert raw not in rendered


def test_continual_factorized_update_can_train_but_fail_strict_l30_court():
    candidate, reference = cloned_pair()
    receipt = train_continual_factorized_update(
        candidate,
        reference,
        [example([1, 11, 12, 13]), example([1, 14, 15, 16])],
        [example([1, 5, 6, 7]), example([1, 8, 9, 10])],
        adaptation_group_sha256=[digest("new-a"), digest("new-b")],
        retention_group_sha256=[digest("old-a"), digest("old-b")],
        config=ContinualFactorizedUpdateConfig(
            epochs=1,
            learning_rate=0.005,
        ),
        court_policy=ContinualLearningPolicy(
            min_retention_groups=2,
            min_adaptation_groups=2,
            max_worst_retention_regression=10.0,
            max_mean_retention_regression=10.0,
            min_mean_adaptation_gain=100.0,
            max_worst_adaptation_regression=10.0,
        ),
    )
    assert receipt.candidate_boundary_changed
    assert receipt.continual_learning["status"] == "BLOCKED"
    assert "adaptation_gain_failed" in receipt.continual_learning["reasons"]


def test_continual_factorized_update_rejects_nonidentical_start_state():
    candidate, reference = cloned_pair()
    with torch.no_grad():
        next(candidate.boundary.module.parameters()).add_(0.1)

    with pytest.raises(ValueError, match="exact reference boundary state"):
        train_continual_factorized_update(
            candidate,
            reference,
            [example([1, 11, 12])],
            [example([1, 5, 6])],
            adaptation_group_sha256=[digest("new-a")],
            retention_group_sha256=[digest("old-a")],
        )


def test_continual_factorized_update_rejects_alias_and_lineage_drift():
    candidate, reference = cloned_pair()

    with pytest.raises(ValueError, match="distinct model objects"):
        train_continual_factorized_update(
            reference,
            reference,
            [example([1, 11, 12])],
            [example([1, 5, 6])],
            adaptation_group_sha256=[digest("new-a")],
            retention_group_sha256=[digest("old-a")],
        )

    with pytest.raises(ValueError, match="adaptation source-group lineage length"):
        train_continual_factorized_update(
            candidate,
            reference,
            [example([1, 11, 12])],
            [example([1, 5, 6])],
            adaptation_group_sha256=[],
            retention_group_sha256=[digest("old-a")],
        )
