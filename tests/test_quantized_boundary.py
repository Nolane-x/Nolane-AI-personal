import pytest

torch=pytest.importorskip("torch")

from nolane_personal.deep_recurrent_cortex import DeepRecurrentCortexConfig, DeepRecurrentStateSpaceCortex
from nolane_personal.factorized_boundary import factorize_standalone_boundary
from nolane_personal.quantized_artifact import load_quantized_model, save_quantized_artifact
from nolane_personal.quantized_boundary import dequantize_rows, quantize_factorized_boundary, quantize_rows
from nolane_personal.standalone_model import StandaloneBoundaryModule, StandaloneNolaneConfig, StandaloneNolaneLM
from nolane_personal.surgery import module_parameter_digest


def source_boundary(*, tied=True, vocab=257, hidden=16):
    torch.manual_seed(7)
    return StandaloneBoundaryModule(
        StandaloneNolaneConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            tie_word_embeddings=tied,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def model(boundary, seed=11):
    cortex=DeepRecurrentStateSpaceCortex(
        boundary.config.hidden_size,
        DeepRecurrentCortexConfig(
            state_dim=8,
            latent_dim=32,
            virtual_steps=3,
            max_virtual_steps=4,
        ),
        seed=seed,
    )
    return StandaloneNolaneLM(boundary,cortex,[0.2]*32)


def test_rowwise_int8_quantization_handles_zero_rows():
    x=torch.tensor([[0.0,0.0,0.0],[1.0,-0.5,0.25]],dtype=torch.float32)
    q,s=quantize_rows(x)
    y=dequantize_rows(q,s)
    assert q.dtype==torch.int8
    assert s.dtype==torch.float32
    assert torch.isfinite(y).all()
    assert torch.equal(y[0],torch.zeros_like(y[0]))
    assert torch.allclose(y[1],x[1],atol=0.01,rtol=0.01)


def test_quantized_tied_boundary_is_small_and_close():
    dense=source_boundary(tied=True)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=13)
    quantized,receipt=quantize_factorized_boundary(factorized,logit_chunk_size=31)
    assert quantized.config.tie_word_embeddings
    assert receipt["storage_ratio"] < 0.45
    assert quantized.storage_bytes() < receipt["source_float_storage_bytes"]*0.45
    assert receipt["input_code_relative_error"] < 0.02
    assert receipt["input_basis_relative_error"] < 0.02
    ids=torch.tensor([[1,4,5,6]])
    with torch.no_grad():
        a=factorized.embed_tokens(ids)
        b=quantized.embed_tokens(ids)
    assert torch.allclose(a,b,atol=0.03,rtol=0.05)


def test_quantized_logits_track_factorized_tied_boundary():
    dense=source_boundary(tied=True,vocab=131,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=17)
    quantized,_=quantize_factorized_boundary(factorized,logit_chunk_size=17)
    hidden=torch.randn(2,5,16)
    with torch.no_grad():
        reference=factorized.logits(hidden)
        candidate=quantized.logits(hidden)
    rel=(candidate-reference).norm()/reference.norm().clamp_min(1e-12)
    assert float(rel)<0.04


def test_quantized_untied_boundary_tracks_factorized_output():
    dense=source_boundary(tied=False,vocab=127,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=19)
    quantized,receipt=quantize_factorized_boundary(factorized,logit_chunk_size=23)
    assert not quantized.config.tie_word_embeddings
    assert quantized.output is not None
    assert receipt["output_code_relative_error"]<0.02
    assert receipt["output_basis_relative_error"]<0.02
    hidden=torch.randn(1,3,16)
    with torch.no_grad():
        a=factorized.logits(hidden)
        b=quantized.logits(hidden)
    rel=(a-b).norm()/a.norm().clamp_min(1e-12)
    assert float(rel)<0.04


def test_quantized_runtime_preserves_prompt_scan_equivalence():
    dense=source_boundary(tied=True,vocab=137,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=23)
    quantized,_=quantize_factorized_boundary(factorized,logit_chunk_size=19)
    m=model(quantized,seed=29)
    ids=torch.tensor([[1,5,6,7]])
    assert m.prompt_scan_equivalent(ids,atol=1e-5,rtol=1e-5)
    out=m.generate(input_ids=ids,max_new_tokens=2,do_sample=False,eos_token_id=None)
    assert out.shape[1]==6


def test_quantized_boundary_is_inference_only():
    dense=source_boundary(tied=True)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=31)
    quantized,_=quantize_factorized_boundary(factorized)
    with pytest.raises(RuntimeError,match="inference-only"):
        quantized.train()


def test_quantized_artifact_roundtrip_and_cortex_digest(tmp_path):
    dense=source_boundary(tied=True,vocab=149,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=37)
    quantized,receipt=quantize_factorized_boundary(factorized,logit_chunk_size=29)
    m=model(quantized,seed=41)
    source_cortex_digest=module_parameter_digest(m.cortex.module)
    manifest=save_quantized_artifact(
        tmp_path,
        m,
        receipt,
        source_meta={
            "checkpoint_sha256":"factorized-source",
            "source_l16_checkpoint_sha256":"l16-source",
            "dataset_fingerprint":"protocol",
        },
        dataset_fingerprint="protocol",
    )
    loaded,meta=load_quantized_model(
        tmp_path/"quantized-nolane.pt",
        [0.2]*32,
        expected_source_factorized_checkpoint_sha256="factorized-source",
        expected_source_l16_checkpoint_sha256="l16-source",
        expected_dataset_fingerprint="protocol",
    )
    assert meta["runtime_requires_qwen_model"] is False
    assert meta["runtime_requires_transformers"] is False
    assert meta["cortex_state_digest"]==source_cortex_digest==manifest["cortex_state_digest"]
    assert meta["boundary_state_digest"]==manifest["boundary_state_digest"]
    ids=torch.tensor([[1,8,9,10]])
    with torch.no_grad():
        a=m.forward(input_ids=ids)
        b=loaded.forward(input_ids=ids)
    assert torch.equal(a.logits,b.logits)
    assert torch.equal(a.state,b.state)
