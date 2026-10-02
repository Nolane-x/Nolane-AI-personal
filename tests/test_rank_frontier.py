import pytest

torch=pytest.importorskip("torch")

from nolane_personal.deep_recurrent_cortex import DeepRecurrentCortexConfig,DeepRecurrentStateSpaceCortex
from nolane_personal.rank_frontier import RankFrontierConfig,search_rank_frontier
from nolane_personal.standalone_model import StandaloneBoundaryModule,StandaloneNolaneConfig,StandaloneNolaneLM
from nolane_personal.surgery import module_parameter_digest


def source_model():
    torch.manual_seed(3)
    boundary=StandaloneBoundaryModule(
        StandaloneNolaneConfig(
            vocab_size=101,
            hidden_size=16,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    cortex=DeepRecurrentStateSpaceCortex(
        16,
        DeepRecurrentCortexConfig(
            latent_dim=32,
            state_dim=8,
            virtual_steps=3,
            max_virtual_steps=4,
        ),
        seed=5,
    )
    return StandaloneNolaneLM(boundary,cortex,[0.2]*32)


def example(tokens):
    ids=list(tokens)
    return (ids,[-100]+[3]*(len(ids)-1),[0.2]*32,1.0)


def test_rank_frontier_descends_and_keeps_cortex_frozen():
    source=source_model()
    before=module_parameter_digest(source.cortex.module)
    train=[example([1,5,6,7]),example([1,8,9,10])]
    dev=[example([1,11,12,13]),example([1,14,15,16])]
    selected,receipt=search_rank_frontier(
        source,train,dev,
        config=RankFrontierConfig(
            ranks=(8,4,2),
            max_dev_nll_regression_per_step=100.0,
            max_dev_nll_regression_vs_l16=100.0,
            min_greedy_token_agreement=0.0,
            max_boundary_parameter_ratio=0.60,
            distill_epochs=1,
            learning_rate=0.01,
            distill_weight=0.1,
        ),
    )
    assert receipt.ranks_attempted==[8,4,2]
    assert receipt.ranks_accepted==[8,4,2]
    assert receipt.selected_rank==2
    ratios=[row["boundary_parameter_ratio"] for row in receipt.stages]
    assert ratios[0]>ratios[1]>ratios[2]
    assert receipt.selected_boundary_parameter_ratio==ratios[-1]
    assert module_parameter_digest(source.cortex.module)==before
    assert module_parameter_digest(selected.cortex.module)==before
    assert all(row["cortex_unchanged"] for row in receipt.stages)


def test_rank_frontier_fails_closed_when_first_rank_is_not_compressed_enough():
    source=source_model()
    train=[example([1,5,6,7])]
    dev=[example([1,8,9,10])]
    with pytest.raises(RuntimeError,match="no low-rank boundary candidate"):
        search_rank_frontier(
            source,train,dev,
            config=RankFrontierConfig(
                ranks=(16,),
                max_dev_nll_regression_per_step=100.0,
                max_dev_nll_regression_vs_l16=100.0,
                min_greedy_token_agreement=0.0,
                max_boundary_parameter_ratio=0.60,
                distill_epochs=1,
            ),
        )


def test_rank_frontier_contract_rejects_non_descending_schedule():
    with pytest.raises(ValueError,match="strictly descending"):
        RankFrontierConfig(ranks=(8,8,4)).validate(hidden_size=16)
