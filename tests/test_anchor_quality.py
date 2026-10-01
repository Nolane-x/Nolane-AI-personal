from nolane_personal.anchor_evaluation import AnchorQualityEvidence,AnchorResourceEvidence,decide_anchor_quality,decide_anchor_resources
from nolane_personal.anchor_promotion import decide_anchor_promotion
def quality(**kw):
 d=dict(test_examples=20,anchor_examples=8,total_layers=28,head_layers=1,tail_layers=1,remaining_qwen_layers=2,stages_completed=4,virtual_steps=8,baseline_nll=2.0,l13_nll=1.65,anchor_cortex_nll=1.66,improvement_vs_base=.34,degradation_vs_l13=.01,anchor_baseline_nll=1.5,anchor_cortex_anchor_nll=1.53,anchor_nll_regression=.03,cached_generation_passed=True,scan_equivalence_passed=True,virtual_depth_effect_passed=True,cortex_parameters=89805,base_model_unchanged=True,base_gradients_seen=0); d.update(kw); return AnchorQualityEvidence(**d)
def resources(**kw):
 d=dict(prompts=8,total_layers=28,remaining_qwen_layers=2,baseline_median_ms=100.,l13_median_ms=70.,anchor_cortex_median_ms=55.,latency_ratio_vs_base=.55,latency_ratio_vs_l13=.786,cached_tokens_per_second=48.,replay_safe_tokens_per_second=30.,cached_speedup_vs_replay=1.6,artifact_bytes=500000,cortex_parameters=89805); d.update(kw); return AnchorResourceEvidence(**d)
def test_quality_requires_thin_shell_and_real_virtual_depth():
 assert decide_anchor_quality(quality()).status=="MINIMAL_ANCHOR_QUALITY_PASS"
 assert "qwen_anchor_shell_too_large" in decide_anchor_quality(quality(remaining_qwen_layers=6)).reasons
 assert "virtual_depth_effect_gate_failed" in decide_anchor_quality(quality(virtual_depth_effect_passed=False)).reasons
 assert "l13_noninferiority_failed" in decide_anchor_quality(quality(degradation_vs_l13=.05)).reasons
def test_resources_and_promotion_bind_identity():
 assert decide_anchor_resources(resources())["status"]=="MINIMAL_ANCHOR_RESOURCE_PASS"
 assert "latency_gain_vs_l13_failed" in decide_anchor_resources(resources(latency_ratio_vs_l13=.95))["reasons"]
 p=decide_anchor_promotion(quality_status="MINIMAL_ANCHOR_QUALITY_PASS",quality_checkpoint_sha256="c",quality_plan_sha256="p",resource_status="MINIMAL_ANCHOR_RESOURCE_PASS",resource_checkpoint_sha256="c",resource_plan_sha256="p"); assert p.status=="MINIMAL_ANCHOR_PROMOTION_PASS"
 b=decide_anchor_promotion(quality_status="MINIMAL_ANCHOR_QUALITY_PASS",quality_checkpoint_sha256="a",quality_plan_sha256="x",resource_status="MINIMAL_ANCHOR_RESOURCE_PASS",resource_checkpoint_sha256="b",resource_plan_sha256="y"); assert "court_checkpoint_mismatch" in b.reasons and "court_plan_mismatch" in b.reasons
