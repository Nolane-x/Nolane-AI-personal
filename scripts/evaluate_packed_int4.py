from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.latent import LatentStore
from nolane_personal.packed_int4_artifact import load_packed_int4_model
from nolane_personal.personal_dataset import encode_chat_example,load_jsonl
from nolane_personal.personal_protocol import examples_for_split,load_protocol,verify_personalization_protocol
from nolane_personal.precision_frontier import PackedInt4QualityEvidence,decide_packed_int4_quality
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.quantized_artifact import load_quantized_model


def sha256_file(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            d.update(c)
    return d.hexdigest()


def encode(tok,examples,latent,max_length):
    rows=[]
    for e in examples:
        ids,labels=encode_chat_example(tok,e,system_prompt=SYSTEM_PROMPT,max_length=max_length)
        rows.append((ids,labels,e.latent or latent,e.weight))
    return rows


def metrics(model,examples):
    import torch
    device=next(model.boundary.module.parameters()).device
    losses=[]; preds=[]
    model.eval()
    with torch.no_grad():
        for ids,labels,latent,weight in examples:
            x=torch.tensor([ids],dtype=torch.long,device=device)
            y=torch.tensor([labels],dtype=torch.long,device=device)
            if latent is not None:
                model.set_latent(latent)
            out=model.forward(input_ids=x,labels=y,state=None)
            losses.append(float(out.loss.detach().cpu())*float(weight))
            preds.append(torch.argmax(out.logits,dim=-1).detach().cpu())
    return sum(losses)/max(1,len(losses)),preds


def agreement(left,right):
    total=equal=0
    for a,b in zip(left,right):
        total+=a.numel()
        equal+=int((a==b).sum())
    return equal/max(1,total)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--factorized",default="runtime-data/l18-rank-frontier/factorized-nolane.pt")
    p.add_argument("--int8",default="runtime-data/l19-quantized/quantized-nolane.pt")
    p.add_argument("--int4",default="runtime-data/l20-packed-int4/packed-int4-nolane.pt")
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
    import torch

    dp=Path(a.dataset)
    examples=load_jsonl(dp)
    protocol=load_protocol(a.protocol)
    verify_personalization_protocol(protocol,dataset_sha256=sha256_file(dp))
    test=examples_for_split(examples,protocol,"test")
    anchor=load_jsonl(a.anchor)
    latent=LatentStore(a.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {a.latent}")

    factorized,fmeta=load_factorized_model(
        a.factorized,latent.values,device=a.device,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    int8,i8meta=load_quantized_model(
        a.int8,latent.values,device=a.device,
        expected_source_factorized_checkpoint_sha256=fmeta["checkpoint_sha256"],
        expected_source_l16_checkpoint_sha256=fmeta["source_l16_checkpoint_sha256"],
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    int4,i4meta=load_packed_int4_model(
        a.int4,latent.values,device=a.device,
        expected_source_factorized_checkpoint_sha256=fmeta["checkpoint_sha256"],
        expected_source_l16_checkpoint_sha256=fmeta["source_l16_checkpoint_sha256"],
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    t=encode(tok,test,latent.values,a.max_length)
    g=encode(tok,anchor,latent.values,a.max_length)

    fnll,fpred=metrics(factorized,t)
    i8nll,i8pred=metrics(int8,t)
    i4nll,i4pred=metrics(int4,t)
    fanll,_=metrics(factorized,g)
    i8anll,_=metrics(int8,g)
    i4anll,_=metrics(int4,g)

    device=next(int4.boundary.module.parameters()).device
    ev=PackedInt4QualityEvidence(
        test_examples=len(t),anchor_examples=len(g),rank=int(int4.boundary.config.rank),
        factorized_nll=fnll,int8_nll=i8nll,int4_nll=i4nll,
        int4_regression_vs_factorized=i4nll-fnll,
        int4_regression_vs_int8=i4nll-i8nll,
        anchor_factorized_nll=fanll,anchor_int8_nll=i8anll,anchor_int4_nll=i4anll,
        anchor_int4_regression_vs_factorized=i4anll-fanll,
        anchor_int4_regression_vs_int8=i4anll-i8anll,
        greedy_agreement_int4_vs_factorized=agreement(fpred,i4pred),
        greedy_agreement_int4_vs_int8=agreement(i8pred,i4pred),
        prompt_scan_equivalence_passed=bool(
            int4.prompt_scan_equivalent(
                torch.tensor([t[0][0]],dtype=torch.long,device=device)
            )
        ) if t else False,
        cortex_digest_equal=(
            i4meta["cortex_state_digest"]==i8meta["cortex_state_digest"]==fmeta["cortex_state_digest"]
        ),
        rank_equal=(
            int(int4.boundary.config.rank)==int(int8.boundary.config.rank)==int(factorized.boundary.config.rank)
        ),
        runtime_requires_qwen_model=False,
        runtime_requires_transformers=False,
    )
    decision=decide_packed_int4_quality(ev)
    result={
        "schema":"NOLANE-L20-PACKED-INT4-QUALITY-EVAL-V1",
        "authority":"EVALUATION_ONLY_UNPROMOTED",
        "packed_int4_checkpoint_sha256":i4meta["checkpoint_sha256"],
        "reference_int8_checkpoint_sha256":i8meta["checkpoint_sha256"],
        "source_factorized_checkpoint_sha256":fmeta["checkpoint_sha256"],
        "source_l16_checkpoint_sha256":fmeta["source_l16_checkpoint_sha256"],
        "decision":decision,
    }
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite evaluation: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if decision["status"]=="PACKED_INT4_QUALITY_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
