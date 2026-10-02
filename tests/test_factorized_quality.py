from nolane_personal.factorized_evaluation import FactorizedQualityEvidence,FactorizedResourceEvidence,decide_factorized_quality,decide_factorized_resources
from nolane_personal.factorized_promotion import decide_factorized_promotion

def q(**kw):
    d=dict(test_examples=20,anchor_examples=8,rank=128,hidden_size=1024,dense_boundary_parameters=155000000,factorized_boundary_parameters=19500000,boundary_parameter_ratio=.126,input_reconstruction_error=.08,output_reconstruction_error=.08,l16_nll=1.7,l17_nll=1.72,nll_regression_vs_l16=.02,anchor_l16_nll=1.5,anchor_l17_nll=1.53,anchor_nll_regression=.03,greedy_token_agreement=.97,prompt_scan_equivalence_passed=True,cortex_digest_equal_to_l16=True,runtime_requires_qwen_model=False,runtime_requires_transformers=False); d.update(kw); return FactorizedQualityEvidence(**d)
def r(**kw):
    d=dict(prompts=8,l16_median_ms=50.,l17_median_ms=52.,latency_ratio_vs_l16=1.04,l16_checkpoint_bytes=620000000,l17_checkpoint_bytes=90000000,checkpoint_ratio=.145,boundary_parameter_ratio=.126,tokens_per_second=30.); d.update(kw); return FactorizedResourceEvidence(**d)

def test_quality_requires_real_compression_and_l16_noninferiority():
    assert decide_factorized_quality(q())["status"]=="FACTORIZED_BOUNDARY_QUALITY_PASS"
    assert "boundary_compression_failed" in decide_factorized_quality(q(boundary_parameter_ratio=.7))["reasons"]
    assert "l16_noninferiority_failed" in decide_factorized_quality(q(nll_regression_vs_l16=.08))["reasons"]
    assert "cortex_digest_drift" in decide_factorized_quality(q(cortex_digest_equal_to_l16=False))["reasons"]

def test_resource_and_promotion_bind_l16_lineage():
    assert decide_factorized_resources(r())["status"]=="FACTORIZED_BOUNDARY_RESOURCE_PASS"
    assert "checkpoint_compression_failed" in decide_factorized_resources(r(checkpoint_ratio=.8))["reasons"]
    p=decide_factorized_promotion(quality_status="FACTORIZED_BOUNDARY_QUALITY_PASS",quality_checkpoint_sha256="c",quality_source_l16_checkpoint_sha256="s",resource_status="FACTORIZED_BOUNDARY_RESOURCE_PASS",resource_checkpoint_sha256="c",resource_source_l16_checkpoint_sha256="s")
    assert p.status=="FACTORIZED_BOUNDARY_PROMOTION_PASS"
    b=decide_factorized_promotion(quality_status="FACTORIZED_BOUNDARY_QUALITY_PASS",quality_checkpoint_sha256="a",quality_source_l16_checkpoint_sha256="x",resource_status="FACTORIZED_BOUNDARY_RESOURCE_PASS",resource_checkpoint_sha256="b",resource_source_l16_checkpoint_sha256="y")
    assert "court_checkpoint_mismatch" in b.reasons and "source_l16_checkpoint_mismatch" in b.reasons
