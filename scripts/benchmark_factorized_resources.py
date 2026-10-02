from __future__ import annotations

import argparse,json,statistics,time
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.factorized_evaluation import FactorizedResourceEvidence,decide_factorized_resources
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.standalone_artifact import load_standalone_model

def encode_prompt(tok,prompt):
    m=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
    try:r=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:r=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    return tok(r,return_tensors="pt")

def timed(fn):
    s=time.perf_counter(); fn(); return (time.perf_counter()-s)*1000

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--l16",default="runtime-data/l16-standalone/standalone-nolane.pt")
    p.add_argument("--factorized",default="runtime-data/l17-factorized-trained/factorized-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--prompts",default="research/personalization-general-anchor.jsonl")
    p.add_argument("--tokenizer",default="models/Qwen3-0.6B")
    p.add_argument("--device",default="cpu")
    p.add_argument("--generation-tokens",type=int,default=8)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    import torch
    from transformers import AutoTokenizer
    latent=LatentStore(a.latent).load()
    l16,m16=load_standalone_model(a.l16,latent.values,device=a.device)
    l17,m17=load_factorized_model(a.factorized,latent.values,device=a.device,expected_source_l16_checkpoint_sha256=m16["checkpoint_sha256"])
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    samples=[{k:v.to(a.device) for k,v in dict(encode_prompt(tok,e.prompt)).items()} for e in load_jsonl(a.prompts)]
    t16=[]; t17=[]; rates=[]
    with torch.no_grad():
        for s in samples:
            t16.append(timed(lambda:l16.forward(input_ids=s["input_ids"],state=None)))
            t17.append(timed(lambda:l17.forward(input_ids=s["input_ids"],state=None)))
            st=time.perf_counter(); out=l17.generate(input_ids=s["input_ids"],max_new_tokens=a.generation_tokens,do_sample=False,eos_token_id=None)
            rates.append(max(1,out.shape[1]-s["input_ids"].shape[1])/max(time.perf_counter()-st,1e-9))
    a16=statistics.median(t16); a17=statistics.median(t17)
    fr=m17["factorization_receipt"]
    ev=FactorizedResourceEvidence(
        prompts=len(samples),l16_median_ms=a16,l17_median_ms=a17,latency_ratio_vs_l16=a17/max(a16,1e-9),
        l16_checkpoint_bytes=Path(a.l16).stat().st_size,l17_checkpoint_bytes=Path(a.factorized).stat().st_size,
        checkpoint_ratio=Path(a.factorized).stat().st_size/max(1,Path(a.l16).stat().st_size),
        boundary_parameter_ratio=float(fr["parameter_ratio"]),tokens_per_second=statistics.median(rates),
    )
    decision=decide_factorized_resources(ev)
    result={"schema":"NOLANE-L17-FACTORIZED-BOUNDARY-RESOURCE-EVAL-V1","authority":"EVALUATION_ONLY_UNPROMOTED","factorized_checkpoint_sha256":m17["checkpoint_sha256"],"source_l16_checkpoint_sha256":m16["checkpoint_sha256"],"decision":decision}
    s=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists(): raise SystemExit(f"refusing to overwrite resource evaluation: {path}")
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(s+"\n",encoding="utf-8")
    print(s); return 0 if decision["status"]=="FACTORIZED_BOUNDARY_RESOURCE_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
