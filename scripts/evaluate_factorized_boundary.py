from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.heldout_group_robustness import HeldoutGroupRobustnessPolicy,assess_group_robustness
from nolane_personal.factorized_evaluation import FactorizedQualityEvidence,FactorizedQualityThresholds,decide_factorized_quality
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example,load_jsonl
from nolane_personal.personal_protocol import examples_for_split,load_protocol,verify_personalization_protocol
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.standalone_artifact import load_standalone_model

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

def metrics(model,examples):
    import torch
    device=next(model.boundary.module.parameters()).device
    losses=[]; preds=[]
    model.eval()
    with torch.no_grad():
        for ids,labels,latent,weight in examples:
            x=torch.tensor([ids],dtype=torch.long,device=device); y=torch.tensor([labels],dtype=torch.long,device=device)
            if latent is not None: model.set_latent(latent)
            o=model.forward(input_ids=x,labels=y,state=None)
            losses.append(float(o.loss.detach().cpu())*float(weight))
            preds.append(torch.argmax(o.logits,dim=-1).detach().cpu())
    return sum(losses)/max(1,len(losses)),preds,losses

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--l16",default="runtime-data/l16-standalone/standalone-nolane.pt")
    p.add_argument("--factorized",default="runtime-data/l17-factorized-trained/factorized-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--dataset",default="runtime-data/personalization.jsonl")
    p.add_argument("--protocol",default="runtime-data/personalization-protocol-v1.json")
    p.add_argument("--anchor",default="research/personalization-general-anchor.jsonl")
    p.add_argument("--tokenizer",default="models/Qwen3-0.6B")
    p.add_argument("--device",default="cpu")
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    from transformers import AutoTokenizer
    dp=Path(a.dataset); examples=load_jsonl(dp); protocol=load_protocol(a.protocol)
    verify_personalization_protocol(protocol,dataset_sha256=sha256_file(dp))
    test=examples_for_split(examples,protocol,"test"); anchor=load_jsonl(a.anchor)
    latent=LatentStore(a.latent).load()
    teacher,tmeta=load_standalone_model(a.l16,latent.values,device=a.device,expected_dataset_fingerprint=protocol["protocol_sha256"])
    student,smeta=load_factorized_model(a.factorized,latent.values,device=a.device,expected_source_l16_checkpoint_sha256=tmeta["checkpoint_sha256"],expected_dataset_fingerprint=protocol["protocol_sha256"])
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    t=encode(tok,test,latent.values,a.max_length); g=encode(tok,anchor,latent.values,a.max_length)
    n16,p16,l16_values=metrics(teacher,t); n17,p17,l17_values=metrics(student,t)
    a16,_,_=metrics(teacher,g); a17,_,_=metrics(student,g)
    total=agree=0
    for x,y in zip(p16,p17):
        total+=x.numel(); agree+=int((x==y).sum())
    fr=smeta["factorization_receipt"]
    ev=FactorizedQualityEvidence(
        test_examples=len(t),anchor_examples=len(g),rank=int(student.boundary.config.rank),hidden_size=int(student.boundary.config.hidden_size),
        dense_boundary_parameters=int(fr["dense_parameters"]),factorized_boundary_parameters=int(fr["factorized_parameters"]),
        boundary_parameter_ratio=float(fr["parameter_ratio"]),input_reconstruction_error=float(fr["input_relative_frobenius_error"]),
        output_reconstruction_error=float(fr["output_relative_frobenius_error"]),l16_nll=n16,l17_nll=n17,nll_regression_vs_l16=n17-n16,
        anchor_l16_nll=a16,anchor_l17_nll=a17,anchor_nll_regression=a17-a16,greedy_token_agreement=agree/max(1,total),
        prompt_scan_equivalence_passed=bool(student.prompt_scan_equivalent(__import__("torch").tensor([t[0][0]],dtype=__import__("torch").long,device=next(student.boundary.module.parameters()).device))) if t else False,
        cortex_digest_equal_to_l16=smeta["cortex_state_digest"]==tmeta["cortex_state_digest"],runtime_requires_qwen_model=False,runtime_requires_transformers=False,
    )
    decision=decide_factorized_quality(ev)
    group_robustness=assess_group_robustness(
        protocol,
        split="test",
        reference_values=l16_values,
        candidate_values=l17_values,
        policy=HeldoutGroupRobustnessPolicy(
            max_worst_group_regression=FactorizedQualityThresholds().max_nll_regression_vs_l16,
        ),
    )
    result={"schema":"NOLANE-L17-FACTORIZED-BOUNDARY-QUALITY-EVAL-V1","authority":"EVALUATION_ONLY_UNPROMOTED","factorized_checkpoint_sha256":smeta["checkpoint_sha256"],"source_l16_checkpoint_sha256":tmeta["checkpoint_sha256"],"decision":decision,"group_robustness":group_robustness}
    s=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists(): raise SystemExit(f"refusing to overwrite evaluation: {path}")
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(s+"\n",encoding="utf-8")
    print(s); return 0 if decision["status"]=="FACTORIZED_BOUNDARY_QUALITY_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
