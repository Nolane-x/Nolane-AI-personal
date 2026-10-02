from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import encode_chat_example,load_jsonl
from nolane_personal.personal_protocol import examples_for_split,load_protocol,verify_personalization_protocol
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.quantized_artifact import load_quantized_model
from nolane_personal.quantized_evaluation import QuantizedQualityEvidence,decide_quantized_quality


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--factorized",default="runtime-data/l18-rank-frontier/factorized-nolane.pt")
    p.add_argument("--quantized",default="runtime-data/l19-quantized/quantized-nolane.pt")
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

    dp=Path(a.dataset)
    examples=load_jsonl(dp)
    protocol=load_protocol(a.protocol)
    verify_personalization_protocol(protocol,dataset_sha256=sha256_file(dp))
    test=examples_for_split(examples,protocol,"test")
    anchor=load_jsonl(a.anchor)
    latent=LatentStore(a.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {a.latent}")

    source,smeta=load_factorized_model(
        a.factorized,
        latent.values,
        device=a.device,
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    candidate,qmeta=load_quantized_model(
        a.quantized,
        latent.values,
        device=a.device,
        expected_source_factorized_checkpoint_sha256=smeta["checkpoint_sha256"],
        expected_source_l16_checkpoint_sha256=smeta["source_l16_checkpoint_sha256"],
        expected_dataset_fingerprint=protocol["protocol_sha256"],
    )
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    t=encode(tok,test,latent.values,a.max_length)
    g=encode(tok,anchor,latent.values,a.max_length)

    source_nll,source_pred=metrics(source,t)
    quant_nll,quant_pred=metrics(candidate,t)
    anchor_source,_=metrics(source,g)
    anchor_quant,_=metrics(candidate,g)
    total=agree=0
    for left,right in zip(source_pred,quant_pred):
        total+=left.numel()
        agree+=int((left==right).sum())

    import torch
    device=next(candidate.boundary.module.parameters()).device
    ev=QuantizedQualityEvidence(
        test_examples=len(t),
        anchor_examples=len(g),
        rank=int(candidate.boundary.config.rank),
        source_factorized_nll=source_nll,
        quantized_nll=quant_nll,
        nll_regression_vs_factorized=quant_nll-source_nll,
        anchor_source_nll=anchor_source,
        anchor_quantized_nll=anchor_quant,
        anchor_nll_regression=anchor_quant-anchor_source,
        greedy_token_agreement=agree/max(1,total),
        prompt_scan_equivalence_passed=bool(
            candidate.prompt_scan_equivalent(
                torch.tensor([t[0][0]],dtype=torch.long,device=device)
            )
        ) if t else False,
        cortex_digest_equal_to_source=qmeta["cortex_state_digest"]==smeta["cortex_state_digest"],
        source_rank_equal=int(candidate.boundary.config.rank)==int(source.boundary.config.rank),
        runtime_requires_qwen_model=False,
        runtime_requires_transformers=False,
    )
    decision=decide_quantized_quality(ev)
    result={
        "schema":"NOLANE-L19-QUANTIZED-QUALITY-EVAL-V1",
        "authority":"EVALUATION_ONLY_UNPROMOTED",
        "quantized_checkpoint_sha256":qmeta["checkpoint_sha256"],
        "source_factorized_checkpoint_sha256":smeta["checkpoint_sha256"],
        "source_l16_checkpoint_sha256":smeta["source_l16_checkpoint_sha256"],
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
    return 0 if decision["status"]=="QUANTIZED_FACTOR_QUALITY_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
