import pytest
torch=pytest.importorskip("torch")

from nolane_personal.deep_recurrent_cortex import DeepRecurrentCortexConfig,DeepRecurrentStateSpaceCortex
from nolane_personal.factorized_artifact import load_factorized_model,save_factorized_artifact
from nolane_personal.factorized_boundary import FactorizedBoundaryConfig,analytical_factorized_boundary_parameters,dense_boundary_parameters,factorize_standalone_boundary
from nolane_personal.factorized_training import FactorizedTrainingConfig,train_factorized_boundary
from nolane_personal.standalone_model import StandaloneBoundaryModule,StandaloneNolaneConfig,StandaloneNolaneLM
from nolane_personal.surgery import module_parameter_digest

def source_boundary(*,tied=False,vocab=73,hidden=16):
    torch.manual_seed(3)
    return StandaloneBoundaryModule(StandaloneNolaneConfig(vocab_size=vocab,hidden_size=hidden,tie_word_embeddings=tied,bos_token_id=1,eos_token_id=2,pad_token_id=0))

def model(boundary,seed=5):
    c=DeepRecurrentStateSpaceCortex(boundary.config.hidden_size,DeepRecurrentCortexConfig(state_dim=8,latent_dim=32,virtual_steps=3,max_virtual_steps=4),seed=seed)
    return StandaloneNolaneLM(boundary,c,[.2]*32)

def ex(tokens):
    ids=list(tokens); return (ids,[-100]+[3]*(len(ids)-1),[.2]*32,1.0)

def test_full_rank_factorization_reconstructs_boundary():
    src=source_boundary(tied=False,vocab=41,hidden=12)
    fac,receipt=factorize_standalone_boundary(src,rank=12,seed=7)
    assert receipt["parameter_ratio"] < 1.0
    assert receipt["input_relative_frobenius_error"] < 1e-5
    assert receipt["output_relative_frobenius_error"] < 1e-5
    assert torch.allclose(fac.reconstructed_input_weight(),src.embed_tokens.weight,atol=1e-5,rtol=1e-5)
    assert torch.allclose(fac.reconstructed_output_weight(),src.lm_head.weight,atol=1e-5,rtol=1e-5)

def test_low_rank_boundary_reduces_parameters_and_tied_reuses_factors():
    src=source_boundary(tied=True,vocab=97,hidden=16)
    fac,receipt=factorize_standalone_boundary(src,rank=4,seed=11)
    assert fac.config.tie_word_embeddings
    assert fac.output_codes is None and fac.output_basis is None
    assert fac.parameter_count()==analytical_factorized_boundary_parameters(fac.config)
    assert fac.parameter_count() < dense_boundary_parameters(fac.config)*0.35
    ids=torch.tensor([[1,5,6]])
    hidden=fac.embed_tokens(ids)
    assert hidden.shape==(1,3,16)
    assert fac.logits(hidden).shape==(1,3,97)
    assert receipt["parameter_ratio"] < 0.35

def test_factorized_runtime_prompt_scan_equivalence():
    src=source_boundary(tied=True,vocab=53,hidden=16)
    fac,_=factorize_standalone_boundary(src,rank=6,seed=13)
    m=model(fac)
    ids=torch.tensor([[1,5,6,7]])
    assert m.prompt_scan_equivalent(ids)
    out=m.generate(input_ids=ids,max_new_tokens=2,do_sample=False,eos_token_id=None)
    assert out.shape[1]==6

def test_distillation_changes_only_factorized_boundary():
    src=source_boundary(tied=True,vocab=47,hidden=16)
    teacher=model(src,seed=17)
    fac,_=factorize_standalone_boundary(src,rank=5,seed=19)
    student=model(fac,seed=23)
    student.cortex.module.load_state_dict(teacher.cortex.module.state_dict())
    before_c=module_parameter_digest(student.cortex.module)
    before_b=module_parameter_digest(student.boundary.module)
    receipt=train_factorized_boundary(
        student,teacher,[ex([1,5,6,7]),ex([1,8,9,10])],
        config=FactorizedTrainingConfig(epochs=2,learning_rate=.02,distill_weight=.25)
    )
    assert receipt.boundary_changed
    assert receipt.boundary_gradients_seen>0
    assert receipt.cortex_unchanged and receipt.cortex_gradients_seen==0
    assert module_parameter_digest(student.cortex.module)==before_c
    assert module_parameter_digest(student.boundary.module)!=before_b

def test_factorized_artifact_roundtrip(tmp_path):
    src=source_boundary(tied=True,vocab=61,hidden=16)
    fac,fr=factorize_standalone_boundary(src,rank=4,seed=29)
    m=model(fac,seed=31)
    manifest=save_factorized_artifact(
        tmp_path,m,fr,
        source_meta={"checkpoint_sha256":"l16-checkpoint","source_l15_checkpoint_sha256":"l15","dataset_fingerprint":"protocol"},
        dataset_fingerprint="protocol",
    )
    loaded,meta=load_factorized_model(
        tmp_path/"factorized-nolane.pt",[.2]*32,device="cpu",
        expected_source_l16_checkpoint_sha256="l16-checkpoint",
        expected_dataset_fingerprint="protocol",
    )
    assert loaded.boundary.parameter_count()==m.boundary.parameter_count()
    assert meta["runtime_requires_qwen_model"] is False
    assert meta["runtime_requires_transformers"] is False
    assert meta["cortex_state_digest"]==manifest["cortex_state_digest"]
    ids=torch.tensor([[1,4,5]])
    with torch.no_grad():
        a=m.forward(input_ids=ids)
        b=loaded.forward(input_ids=ids)
    assert torch.allclose(a.logits,b.logits,atol=1e-6,rtol=1e-6)
    assert torch.allclose(a.state,b.state,atol=1e-6,rtol=1e-6)
