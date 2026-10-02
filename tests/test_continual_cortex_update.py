import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from nolane_personal.continual_cortex_update import (
    ContinualCortexUpdateConfig,
    train_continual_cortex_update,
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
    torch.manual_seed(101)
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


def model(boundary, seed=103):
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
    reference_boundary, _ = factorize_standalone_boundary(
        source,
        rank=5,
        seed=107,
    )
    candidate_boundary, _ = factorize_standalone_boundary(
        source,
        rank=5,
        seed=107,
    )
    reference = model(reference_boundary)
    candidate = model(candidate_boundary)
    candidate.boundary.module.load_state_dict(
        reference.boundary.module.state_dict()
    )
    candidate.cortex.module.load_state_dict(
        reference.cortex.module.state_dict()
    )
    return candidate, reference


def example(tokens, *, latent_value=0.2):
    ids = list(tokens)
    return (
        ids,
        [-100] + [3] * (len(ids) - 1),
        [latent_value] * 32,
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


def train(candidate, reference, *, policy=None):
    return train_continual_cortex_update(
        candidate,
        reference,
        [
            example([1, 11, 12, 13], latent_value=0.4),
            example([1, 14, 15, 16], latent_value=-0.3),
        ],
        [
            example([1, 5, 6, 7]),
            example([1, 8, 9, 10]),
        ],
        retention_eval_examples=[
            example([1, 17, 18, 19]),
            example([1, 20, 21, 22]),
        ],
        adaptation_eval_examples=[
            example([1, 23, 24, 25], latent_value=0.4),
            example([1, 26, 27, 28], latent_value=-0.3),
        ],
        adaptation_group_sha256=[
            digest("new-a"),
            digest("new-b"),
        ],
        retention_group_sha256=[
            digest("old-a"),
            digest("old-b"),
        ],
        config=ContinualCortexUpdateConfig(
            epochs=2,
            learning_rate=0.01,
            retention_distill_weight=1.0,
            cortex_anchor_weight=0.01,
        ),
        court_policy=policy or permissive_policy(),
    )


def test_cortex_update_changes_only_candidate_recurrent_cortex():
    candidate, reference = cloned_pair()
    candidate_boundary_before = module_parameter_digest(
        candidate.boundary.module
    )
    reference_boundary_before = module_parameter_digest(
        reference.boundary.module
    )
    reference_cortex_before = module_parameter_digest(
        reference.cortex.module
    )

    receipt = train(candidate, reference)

    assert receipt.candidate_cortex_changed
    assert receipt.candidate_cortex_gradients_seen > 0
    assert receipt.candidate_boundary_unchanged
    assert receipt.candidate_boundary_gradients_seen == 0
    assert receipt.reference_boundary_unchanged
    assert receipt.reference_cortex_unchanged
    assert receipt.continual_learning["status"] == "PASS"

    assert (
        module_parameter_digest(candidate.boundary.module)
        == candidate_boundary_before
    )
    assert (
        module_parameter_digest(reference.boundary.module)
        == reference_boundary_before
    )
    assert (
        module_parameter_digest(reference.cortex.module)
        == reference_cortex_before
    )
    assert (
        receipt.candidate_model_state_sha256_before
        != receipt.candidate_model_state_sha256_after
    )


def test_cortex_update_uses_heldout_l30_court_not_optimizer_success():
    candidate, reference = cloned_pair()
    receipt = train(
        candidate,
        reference,
        policy=ContinualLearningPolicy(
            min_retention_groups=2,
            min_adaptation_groups=2,
            max_worst_retention_regression=10.0,
            max_mean_retention_regression=10.0,
            min_mean_adaptation_gain=100.0,
            max_worst_adaptation_regression=10.0,
        ),
    )
    assert receipt.candidate_cortex_changed
    assert receipt.optimizer_steps > 0
    assert receipt.continual_learning["status"] == "BLOCKED"
    assert "adaptation_gain_failed" in receipt.continual_learning["reasons"]


def test_cortex_update_receipt_does_not_leak_source_group_hashes():
    candidate, reference = cloned_pair()
    receipt = train(candidate, reference)
    rendered = json.dumps(receipt.to_dict(), sort_keys=True)
    for raw in (
        digest("new-a"),
        digest("new-b"),
        digest("old-a"),
        digest("old-b"),
    ):
        assert raw not in rendered


def test_cortex_update_rejects_boundary_start_mismatch():
    candidate, reference = cloned_pair()
    with torch.no_grad():
        next(candidate.boundary.module.parameters()).add_(0.1)

    with pytest.raises(ValueError, match="exact reference boundary state"):
        train(candidate, reference)


def test_cortex_update_rejects_cortex_start_mismatch():
    candidate, reference = cloned_pair()
    with torch.no_grad():
        next(candidate.cortex.module.parameters()).add_(0.1)

    with pytest.raises(ValueError, match="exact reference cortex state"):
        train(candidate, reference)


def test_cortex_update_rejects_alias_and_lineage_drift():
    candidate, reference = cloned_pair()
    with pytest.raises(ValueError, match="distinct model objects"):
        train_continual_cortex_update(
            reference,
            reference,
            [example([1, 11, 12])],
            [example([1, 5, 6])],
            retention_eval_examples=[example([1, 8, 9])],
            adaptation_eval_examples=[example([1, 14, 15])],
            adaptation_group_sha256=[digest("new-a")],
            retention_group_sha256=[digest("old-a")],
        )

    with pytest.raises(
        ValueError,
        match="adaptation source-group lineage length",
    ):
        train_continual_cortex_update(
            candidate,
            reference,
            [example([1, 11, 12])],
            [example([1, 5, 6])],
            retention_eval_examples=[example([1, 8, 9])],
            adaptation_eval_examples=[example([1, 14, 15])],
            adaptation_group_sha256=[],
            retention_group_sha256=[digest("old-a")],
        )


def test_cortex_anchor_weight_validation():
    with pytest.raises(
        ValueError,
        match="cortex_anchor_weight must be non-negative",
    ):
        ContinualCortexUpdateConfig(
            cortex_anchor_weight=-0.1,
        ).validate()
