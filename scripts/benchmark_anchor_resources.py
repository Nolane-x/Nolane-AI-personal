from __future__ import annotations
import argparse,json,statistics,time
from pathlib import Path
from nolane_personal.anchor_artifact import build_anchor_model
from nolane_personal.anchor_evaluation import AnchorResourceEvidence,decide_anchor_resources
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import QwenCortex,SYSTEM_PROMPT
from nolane_personal.scaffold_artifact import build_scaffold_model
from nolane_personal.surgery_candidate import load_model_lock,model_lock_fingerprint
ROOT=Path(__file__).resolve().parents[1]
def sync(torch,d):
    if str(d).startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize()
def enc(tok,prompt):
    m=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
    try:r=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:r=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    return tok(r,return_tensors="pt")
def timed(torch,fn,d):
    sync(torch,d); s=time.perf_counter(); fn(); sync(torch,d); return (time.perf_counter()-s)*1000
def tps(torch,m,sample,cached,n,pad):
    m.reset_state(); sync(torch,next(m.model.parameters()).device); s=time.perf_counter()
    out=(m.generate_cached if cached else m.generate_replay_safe)(**sample,max_new_tokens=n,do_sample=False,pad_token_id=pad)
    sync(torch,next(m.model.parameters()).device); return max(1,out.shape[1]-sample["input_ids"].shape[1])/max(time.perf_counter()-s,1e-9)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-lock",default=str(ROOT/"model.lock.json")); p.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B")); p.add_argument("--anchor-cortex",default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt"); p.add_argument("--l13-scaffold",default="runtime-data/l13-shrinking-scaffold/shrinking-qwen-scaffold.pt"); p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json"); p.add_argument("--prompts",default=str(ROOT/"research/personalization-general-anchor.jsonl")); p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]); p.add_argument("--generation-tokens",type=int,default=8); p.add_argument("--output",default=None); a=p.parse_args()
    import torch
    latent=LatentStore(a.latent).load(); fp=model_lock_fingerprint(load_model_lock(a.model_lock)); q=QwenCortex(a.model,device=a.device)
    l14,m14=build_anchor_model(q.model,a.anchor_cortex,latent.values,expected_base_model_fingerprint=fp); l13,_=build_scaffold_model(q.model,a.l13_scaffold,latent.values,expected_base_model_fingerprint=fp)
    samples=[{k:v.to(q.device) for k,v in dict(enc(q.tokenizer,e.prompt)).items()} for e in load_jsonl(a.prompts)]; pad=q.tokenizer.eos_token_id or 0
    bt=[]; t13=[]; t14=[]; ct=[]; rt=[]
    with torch.inference_mode():
      for s in samples:
        bt.append(timed(torch,lambda:q.model(**s,use_cache=False),q.device)); t13.append(timed(torch,lambda:l13.forward(**s,use_cache=False),q.device)); t14.append(timed(torch,lambda:l14.forward(**s,use_cache=False),q.device)); ct.append(tps(torch,l14,s,True,a.generation_tokens,pad)); rt.append(tps(torch,l14,s,False,a.generation_tokens,pad))
    b=statistics.median(bt); x13=statistics.median(t13); x14=statistics.median(t14); c=statistics.median(ct); r=statistics.median(rt); tr=m14.get("training_receipt") or {}
    ev=AnchorResourceEvidence(prompts=len(samples),total_layers=int(q.model.config.num_hidden_layers),remaining_qwen_layers=int(tr.get("remaining_qwen_layers",0)),baseline_median_ms=b,l13_median_ms=x13,anchor_cortex_median_ms=x14,latency_ratio_vs_base=x14/max(b,1e-9),latency_ratio_vs_l13=x14/max(x13,1e-9),cached_tokens_per_second=c,replay_safe_tokens_per_second=r,cached_speedup_vs_replay=c/max(r,1e-9),artifact_bytes=Path(a.anchor_cortex).stat().st_size,cortex_parameters=l14.trainable_parameter_count())
    dec=decide_anchor_resources(ev); result={"schema":"NOLANE-L14-MINIMAL-ANCHOR-RESOURCE-EVAL-V1","authority":"EVALUATION_ONLY_UNPROMOTED","anchor_checkpoint_sha256":m14["checkpoint_sha256"],"plan_sha256":m14["plan_sha256"],"decision":dec}; s=json.dumps(result,indent=2,sort_keys=True)
    if a.output: Path(a.output).write_text(s+"\n",encoding="utf-8")
    print(s); return 0 if dec["status"]=="MINIMAL_ANCHOR_RESOURCE_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
