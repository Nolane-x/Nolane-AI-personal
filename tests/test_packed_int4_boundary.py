import pytest

torch=pytest.importorskip("torch")

from nolane_personal.deep_recurrent_cortex import DeepRecurrentCortexConfig, DeepRecurrentStateSpaceCortex
from nolane_personal.factorized_boundary import factorize_standalone_boundary
from nolane_personal.packed_int4_artifact import load_packed_int4_model, save_packed_int4_artifact
from nolane_personal.packed_int4_boundary import (
    PackedInt4BoundaryConfig,
    analytical_int8_storage_bytes,
    analytical_packed_int4_storage_bytes,
    dequantize_packed_rows,
    pack_factorized_boundary,
    quantize_pack_rows,
    unpack_rows,
)
from nolane_personal.standalone_model import StandaloneBoundaryModule, StandaloneNolaneConfig, StandaloneNolaneLM
from nolane_personal.surgery import module_parameter_digest


def source_boundary(*, tied=True, vocab=257, hidden=16):
    torch.manual_seed(17)
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


def model(boundary, seed=19):
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


def test_pack_unpack_roundtrip_logical_int4_even_and_odd_width():
    for width in (4,5):
        x=torch.tensor([
            [(-1.0 + 2.0*i/max(1,width-1)) for i in range(width)],
            [0.0 for _ in range(width)],
        ],dtype=torch.float32)
        packed,scales,columns=quantize_pack_rows(x)
        assert columns==width
        assert packed.dtype==torch.uint8
        assert packed.shape[1]==(width+1)//2
        logical=unpack_rows(packed,width)
        assert logical.min()>=-7 and logical.max()<=7
        y=dequantize_packed_rows(packed,scales,width)
        assert y.shape==x.shape
        assert torch.isfinite(y).all()
        assert torch.equal(y[1],torch.zeros_like(y[1]))


def test_packed_int4_realistically_halves_int8_factor_storage():
    cfg=PackedInt4BoundaryConfig(
        vocab_size=151936,
        hidden_size=1024,
        rank=128,
        tie_word_embeddings=True,
    )
    int8=analytical_int8_storage_bytes(cfg)
    int4=analytical_packed_int4_storage_bytes(cfg)
    assert int4/int8 < 0.55


def test_packed_int4_tied_boundary_runs_and_compresses():
    dense=source_boundary(tied=True)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=23)
    packed,receipt=pack_factorized_boundary(factorized,logit_chunk_size=31)
    assert packed.config.tie_word_embeddings
    assert receipt["storage_ratio_vs_int8"] < 0.70
    ids=torch.tensor([[1,4,5,6]])
    hidden=packed.embed_tokens(ids)
    assert hidden.shape==(1,4,16)
    logits=packed.logits(hidden)
    assert logits.shape==(1,4,257)


def test_packed_int4_untied_boundary_runs():
    dense=source_boundary(tied=False,vocab=127,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=29)
    packed,receipt=pack_factorized_boundary(factorized,logit_chunk_size=19)
    assert packed.output is not None
    assert receipt["output_code_relative_error"] < 0.15
    assert receipt["output_basis_relative_error"] < 0.15
    hidden=torch.randn(1,3,16)
    assert packed.logits(hidden).shape==(1,3,127)


def test_packed_int4_prompt_scan_and_generation():
    dense=source_boundary(tied=True,vocab=137,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=31)
    packed,_=pack_factorized_boundary(factorized,logit_chunk_size=17)
    m=model(packed,seed=37)
    ids=torch.tensor([[1,5,6,7]])
    assert m.prompt_scan_equivalent(ids,atol=1e-5,rtol=1e-5)
    out=m.generate(input_ids=ids,max_new_tokens=2,do_sample=False,eos_token_id=None)
    assert out.shape[1]==6


def test_packed_int4_is_inference_only():
    dense=source_boundary(tied=True)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=41)
    packed,_=pack_factorized_boundary(factorized)
    with pytest.raises(RuntimeError,match="inference-only"):
        packed.train()


def test_packed_int4_artifact_roundtrip(tmp_path):
    dense=source_boundary(tied=True,vocab=149,hidden=16)
    factorized,_=factorize_standalone_boundary(dense,rank=8,seed=43)
    packed,receipt=pack_factorized_boundary(factorized,logit_chunk_size=29)
    m=model(packed,seed=47)
    cortex_digest=module_parameter_digest(m.cortex.module)
    manifest=save_packed_int4_artifact(
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
    loaded,meta=load_packed_int4_model(
        tmp_path/"packed-int4-nolane.pt",
        [0.2]*32,
        expected_source_factorized_checkpoint_sha256="factorized-source",
        expected_source_l16_checkpoint_sha256="l16-source",
        expected_dataset_fingerprint="protocol",
    )
    assert meta["cortex_state_digest"]==cortex_digest==manifest["cortex_state_digest"]
    assert meta["boundary_state_digest"]==manifest["boundary_state_digest"]
    ids=torch.tensor([[1,8,9,10]])
    with torch.no_grad():
        a=m.forward(input_ids=ids)
        b=loaded.forward(input_ids=ids)
    assert torch.equal(a.logits,b.logits)
    assert torch.equal(a.state,b.state)
