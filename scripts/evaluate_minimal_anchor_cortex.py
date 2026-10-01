from __future__ import annotations
import argparse, hashlib, json
from dataclasses import asdict
from pathlib import Path
from nolane_personal.anchor_artifact import build_anchor_model
from nolane_personal.anchor_evaluation import AnchorQualityEvidence, decide_anchor_quality
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example, load_jsonl
from nolane_personal.personal_protocol import examples_for_split, load_protocol, verify_personalization_protocol
from nolane_personal.qwen import QwenCortex, SYSTEM_PROMPT
from nolane_personal.scaffold_artifact import build_scaffold_model
from nolane_personal.state_space_training import mean_encoded_nll
from nolane_personal.surgery import parameter_guard_snapshot, resolve_transformer_layers
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint
ROOT=Path(__file__).resolve().parents[1]
def sha256_file(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): d.update(c)
    return d.hexdigest()
def encode(tok,examples,latent,max_length):
    out=[]
    for e in examples:
        ids,labels=encode_chat_example(tok,e,system_prompt=SYSTEM_PROMPT,max_length=max_length)
        out.append((ids,labels,e.latent or latent,e.weight))
    return out
def capture_input(model,ids,start):
    import torch
    layers=resolve_transformer_layers(model); box={}
    def hook(_m,args,kwargs):
        h=args[0] if args else kwargs.get("hidden_states"); box["h"]=h.detach()
    hd=layers[int(start)].register_forward_pre_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad(): model(input_ids=ids,use_cache=False)
    finally: hd.remove()
    return box["h"]
def contract_check(model,ids,latent):
    import torch
    device=next(model.model.parameters()).device
    t=torch.tensor([ids],dtype=torch.long,device=device)
    h=capture_input(model.model,t,model.config.region.start)
    with torch.no_grad():
        full,fs,_=model.cortex.scan(h,latent,state=None)
        pieces=[]; state=None
        for i in range(h.shape[1]):
            p,state,_=model.cortex.scan(h[:,i:i+1,:],latent,state=state); pieces.append(p)
        inc=torch.cat(pieces,dim=1)
        shallow,ss,_=model.cortex.scan(h,latent,state=None,virtual_steps=1)
    scan_ok=bool(torch.allclose(full,inc,atol=1e-5,rtol=1e-5) and torch.allclose(fs,state,atol=1e-5,rtol=1e-5))
    depth_ok=bool(not torch.allclose(full,shallow,atol=1e-6,rtol=1e-6) and not torch.allclose(fs,ss,atol=1e-6,rtol=1e-6))
    return scan_ok,depth_ok
def cached_check(model,ids,pad):
    import torch
    d=next(model.model.parameters()).device; t=torch.tensor([ids],dtype=torch.long,device=d)
    model.reset_state(); a=model.generate_cached(input_ids=t,max_new_tokens=2,do_sample=False,pad_token_id=pad)
    model.reset_state(); b=model.generate_replay_safe(input_ids=t,max_new_tokens=2,do_sample=False,pad_token_id=pad)
    return bool(torch.equal(a,b))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-lock",default=str(ROOT/"model.lock.json")); p.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B"))
    p.add_argument("--anchor-cortex",default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt"); p.add_argument("--l13-scaffold",default="runtime-data/l13-shrinking-scaffold/shrinking-qwen-scaffold.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json"); p.add_argument("--dataset",default="runtime-data/personalization.jsonl"); p.add_argument("--protocol",default="runtime-data/personalization-protocol-v1.json")
    p.add_argument("--anchor",default=str(ROOT/"research/personalization-general-anchor.jsonl")); p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]); p.add_argument("--max-length",type=int,default=384); p.add_argument("--output",default=None); a=p.parse_args()
    dp=Path(a.dataset); ex=load_jsonl(dp); protocol=load_protocol(a.protocol); verify_personalization_protocol(protocol,dataset_sha256=sha256_file(dp))
    test_ex=examples_for_split(ex,protocol,"test"); anchor_ex=load_jsonl(a.anchor); latent=LatentStore(a.latent).load()
    if latent is None: raise SystemExit("persistent latent not found")
    fp=model_lock_fingerprint(load_model_lock(a.model_lock)); q=QwenCortex(a.model,device=a.device)
    l14,m14=build_anchor_model(q.model,a.anchor_cortex,latent.values,expected_base_model_fingerprint=fp,expected_dataset_fingerprint=protocol["protocol_sha256"])
    l13,m13=build_scaffold_model(q.model,a.l13_scaffold,latent.values,expected_base_model_fingerprint=fp,expected_dataset_fingerprint=protocol["protocol_sha256"])
    test=encode(q.tokenizer,test_ex,latent.values,a.max_length); anchor=encode(q.tokenizer,anchor_ex,latent.values,a.max_length)
    before=parameter_guard_snapshot(q.model); base=mean_encoded_nll(l14,test,cortex_enabled=False); n13=mean_encoded_nll(l13,test,cortex_enabled=True); n14=mean_encoded_nll(l14,test,cortex_enabled=True)
    ab=mean_encoded_nll(l14,anchor,cortex_enabled=False); aa=mean_encoded_nll(l14,anchor,cortex_enabled=True)
    pad=q.tokenizer.eos_token_id or 0; cached=cached_check(l14,test[0][0],pad) if test else False; scan,depth=contract_check(l14,test[0][0],test[0][2] if test and test[0][2] is not None else latent.values) if test else (False,False)
    after=parameter_guard_snapshot(q.model); tr=m14.get("training_receipt") or {}
    ev=AnchorQualityEvidence(test_examples=len(test),anchor_examples=len(anchor),total_layers=int(q.model.config.num_hidden_layers),head_layers=int(tr.get("head_layers",0)),tail_layers=int(tr.get("tail_layers",0)),remaining_qwen_layers=int(tr.get("remaining_qwen_layers",0)),stages_completed=int(tr.get("stages_accepted",0)),virtual_steps=int(l14.cortex.config.virtual_steps),baseline_nll=base,l13_nll=n13,anchor_cortex_nll=n14,improvement_vs_base=base-n14,degradation_vs_l13=n14-n13,anchor_baseline_nll=ab,anchor_cortex_anchor_nll=aa,anchor_nll_regression=aa-ab,cached_generation_passed=cached,scan_equivalence_passed=scan,virtual_depth_effect_passed=depth,cortex_parameters=l14.trainable_parameter_count(),base_model_unchanged=before==after,base_gradients_seen=sum(1 for x in q.model.parameters() if x.grad is not None))
    dec=decide_anchor_quality(ev); result={"schema":"NOLANE-L14-MINIMAL-ANCHOR-QUALITY-EVAL-V1","authority":"EVALUATION_ONLY_UNPROMOTED","anchor_checkpoint_sha256":m14["checkpoint_sha256"],"l13_checkpoint_sha256":m13["checkpoint_sha256"],"plan_sha256":m14["plan_sha256"],"decision":asdict(dec)}
    s=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)
    if a.output: Path(a.output).write_text(s+"\n",encoding="utf-8")
    print(s); return 0 if dec.status=="MINIMAL_ANCHOR_QUALITY_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
