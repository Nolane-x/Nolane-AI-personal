import json
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen3Config,Qwen3ForCausalLM
from nolane_personal.anchor_artifact import load_anchor_artifact,save_anchor_artifact
from nolane_personal.anchor_plan import AnchorPlanConfig,build_anchor_plan,verify_anchor_plan
from nolane_personal.anchor_training import train_anchor_stages
from nolane_personal.deep_recurrent_cortex import DeepRecurrentCortexConfig,DeepRecurrentStateSpaceCortex,analytical_deep_recurrent_parameter_count
from nolane_personal.island_plan import IslandSensitivity
from nolane_personal.state_space_model import StateSpaceModelConfig,StateSpacePersonalModel
from nolane_personal.state_space_region import CortexRegion,StateSpaceRegionSession
from nolane_personal.state_space_training import StateSpaceTrainingConfig
from nolane_personal.surgery import module_parameter_digest

def qwen(n=12):
 return Qwen3ForCausalLM(Qwen3Config(vocab_size=67,hidden_size=32,intermediate_size=64,num_hidden_layers=n,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=96,bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=True))
def ex(tokens,latent=.3):
 ids=list(tokens); return (ids,[-100]+[3]*(len(ids)-1),[latent]*32,1.0)
def sens(start,end,score):
 w=end-start+1; return IslandSensitivity(start=start,end=end,width=w,residual_rms_ratio=score*w*.8,cosine_change=score*w*.2,transformation_score=score*w,score_per_layer=score)
def plan():
 return build_anchor_plan(total_layers=12,sensitivity=[sens(1,9,.01),sens(2,10,.2),sens(1,10,.02)],base_model_fingerprint="tiny",dataset_fingerprint="protocol",config=AnchorPlanConfig(stage_remaining_fractions=(.25,),target_remaining_layers=2,min_head_layers=1,min_tail_layers=1))

def test_exact_count_scan_equivalence_and_virtual_depth_effect():
 torch.manual_seed(3); c=DeepRecurrentCortexConfig(); m=DeepRecurrentStateSpaceCortex(32,c,seed=5)
 assert m.parameter_count()==analytical_deep_recurrent_parameter_count(32,c)==7469
 assert analytical_deep_recurrent_parameter_count(1024,c)==89805
 h=torch.randn(1,5,32); z=[.2]*32
 full,fs,t=m.scan(h,z); pieces=[]; state=None
 for i in range(h.shape[1]):
  p,state,_=m.scan(h[:,i:i+1,:],z,state=state); pieces.append(p)
 inc=torch.cat(pieces,dim=1)
 shallow,ss,_=m.scan(h,z,virtual_steps=1)
 assert torch.allclose(full,inc,atol=1e-6,rtol=1e-6)
 assert torch.allclose(fs,state,atol=1e-6,rtol=1e-6)
 assert not torch.allclose(full,shallow)
 assert not torch.allclose(fs,ss)
 assert t["virtual_microsteps"]==5*c.virtual_steps

def test_one_call_replaces_ten_of_twelve_qwen_blocks():
 torch.manual_seed(7); model=qwen().eval(); cortex=DeepRecurrentStateSpaceCortex(32,seed=9)
 calls={i:0 for i in range(1,11)}; originals={}
 for i in calls:
  layer=model.model.layers[i]; originals[i]=layer.forward
  def counted(*args,_i=i,_f=layer.forward,**kwargs):
   calls[_i]+=1; return _f(*args,**kwargs)
  layer.forward=counted
 scans={"n":0}; oscan=cortex.scan
 def scan(*args,**kwargs): scans["n"]+=1; return oscan(*args,**kwargs)
 cortex.scan=scan
 try:
  with StateSpaceRegionSession(model,cortex,[.2]*32,region=CortexRegion(1,10)) as s:
   out=model(input_ids=torch.tensor([[1,5,7,9]]),use_cache=False)
   assert out.logits.shape==(1,4,67)
   assert all(v==0 for v in calls.values())
   assert scans["n"]==1 and s.cortex_calls==1
   assert all(v==1 for v in s.identity_calls.values())
 finally:
  cortex.scan=oscan
  for i,f in originals.items(): model.model.layers[i].forward=f

def test_anchor_plan_reaches_one_head_one_tail_and_is_fail_closed():
 p=plan(); verify_anchor_plan(p,base_model_fingerprint="tiny",dataset_fingerprint="protocol")
 assert p["target"]["head_layers"]==1 and p["target"]["tail_layers"]==1
 assert p["target"]["remaining_qwen_layers"]==2
 assert p["target"]["region"]=={"start":1,"end":10}
 bad=json.loads(json.dumps(p)); bad["target"]["remaining_qwen_layers"]=3
 with pytest.raises(ValueError,match="digest mismatch"): verify_anchor_plan(bad,base_model_fingerprint="tiny",dataset_fingerprint="protocol")

def test_cached_and_replay_generation_match_with_minimal_anchors():
 torch.manual_seed(11); base=qwen().eval(); cortex=DeepRecurrentStateSpaceCortex(32,DeepRecurrentCortexConfig(initial_gate=.08),seed=13)
 model=StateSpacePersonalModel(base,cortex,[.2]*32,config=StateSpaceModelConfig(region=CortexRegion(1,10),carry_recurrent_state=False))
 ids=torch.tensor([[1,5,6]])
 a=model.generate_cached(input_ids=ids,max_new_tokens=2,do_sample=False,pad_token_id=0)
 model.reset_state(); b=model.generate_replay_safe(input_ids=ids,max_new_tokens=2,do_sample=False,pad_token_id=0)
 assert torch.equal(a,b)
 assert model.last_identity_calls[10]>=2

def test_multistage_anchor_training_and_artifact_roundtrip(tmp_path):
 torch.manual_seed(17); base=qwen().eval(); cortex=DeepRecurrentStateSpaceCortex(32,DeepRecurrentCortexConfig(initial_gate=.08),seed=19); p=plan(); first=p["stages"][0]["region"]
 model=StateSpacePersonalModel(base,cortex,[.3]*32,config=StateSpaceModelConfig(region=CortexRegion(first["start"],first["end"])))
 train=[ex([1,5,6,7,8,9]),ex([1,10,11,12,13,14])]; dev=[ex([1,15,16,17,18,19]),ex([1,20,21,22,23,24])]
 before=module_parameter_digest(cortex.module)
 r=train_anchor_stages(model,train,dev,p,config=StateSpaceTrainingConfig(distill_epochs_per_stage=1,task_epochs_per_stage=2,learning_rate=.02,distill_learning_rate=.015,max_dev_regression=100.0))
 assert r.stages_accepted==len(p["stages"]); assert r.remaining_qwen_layers==2; assert r.head_layers==1 and r.tail_layers==1
 assert r.base_model_unchanged and r.base_gradients_seen==0 and r.cortex_gradients_seen>0 and module_parameter_digest(cortex.module)!=before
 manifest=save_anchor_artifact(tmp_path,model,r,p,base_model_fingerprint="tiny",dataset_fingerprint="protocol")
 loaded,cfg,meta=load_anchor_artifact(tmp_path/"minimal-qwen-anchor-cortex.pt",expected_base_model_fingerprint="tiny",expected_hidden_size=32,expected_dataset_fingerprint="protocol")
 assert loaded.parameter_count()==cortex.parameter_count(); assert cfg.region==CortexRegion(1,10); assert meta["plan_sha256"]==p["plan_sha256"]; assert meta["cortex_state_digest"]==manifest["cortex_state_digest"]
