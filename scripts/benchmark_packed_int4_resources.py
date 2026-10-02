from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.packed_int4_artifact import load_packed_int4_model
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.precision_frontier import PackedInt4ResourceEvidence,decide_packed_int4_resources
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.quantized_artifact import load_quantized_model


def encode_prompt(tok,prompt):
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
    try:
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError:
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return tok(rendered,return_tensors="pt")


def timed(fn):
    start=time.perf_counter()
    fn()
    return (time.perf_counter()-start)*1000.0


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--int8",default="runtime-data/l19-quantized/quantized-nolane.pt")
    p.add_argument("--int4",default="runtime-data/l20-packed-int4/packed-int4-nolane.pt")
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
    if latent is None:
        raise SystemExit(f"persistent latent not found: {a.latent}")
    int8,i8meta=load_quantized_model(a.int8,latent.values,device=a.device)
    int4,i4meta=load_packed_int4_model(
        a.int4,latent.values,device=a.device,
        expected_source_factorized_checkpoint_sha256=i8meta["source_factorized_checkpoint_sha256"],
        expected_source_l16_checkpoint_sha256=i8meta["source_l16_checkpoint_sha256"],
    )
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    samples=[
        {k:v.to(a.device) for k,v in dict(encode_prompt(tok,e.prompt)).items()}
        for e in load_jsonl(a.prompts)
    ]
    i8times=[]; i4times=[]; rates=[]
    with torch.no_grad():
        for sample in samples:
            i8times.append(timed(lambda:int8.forward(input_ids=sample["input_ids"],state=None)))
            i4times.append(timed(lambda:int4.forward(input_ids=sample["input_ids"],state=None)))
            start=time.perf_counter()
            out=int4.generate(
                input_ids=sample["input_ids"],
                max_new_tokens=a.generation_tokens,
                do_sample=False,
                eos_token_id=None,
            )
            rates.append(
                max(1,out.shape[1]-sample["input_ids"].shape[1])
                / max(time.perf_counter()-start,1e-9)
            )
    i8median=statistics.median(i8times)
    i4median=statistics.median(i4times)
    ev=PackedInt4ResourceEvidence(
        prompts=len(samples),
        int8_median_ms=i8median,int4_median_ms=i4median,
        latency_ratio_vs_int8=i4median/max(i8median,1e-9),
        int8_checkpoint_bytes=Path(a.int8).stat().st_size,
        int4_checkpoint_bytes=Path(a.int4).stat().st_size,
        checkpoint_ratio_vs_int8=Path(a.int4).stat().st_size/max(1,Path(a.int8).stat().st_size),
        int8_boundary_storage_bytes=int8.boundary.storage_bytes(),
        int4_boundary_storage_bytes=int4.boundary.storage_bytes(),
        boundary_storage_ratio_vs_int8=int4.boundary.storage_bytes()/max(1,int8.boundary.storage_bytes()),
        int4_tokens_per_second=statistics.median(rates),
    )
    decision=decide_packed_int4_resources(ev)
    result={
        "schema":"NOLANE-L20-PACKED-INT4-RESOURCE-EVAL-V1",
        "authority":"EVALUATION_ONLY_UNPROMOTED",
        "packed_int4_checkpoint_sha256":i4meta["checkpoint_sha256"],
        "reference_int8_checkpoint_sha256":i8meta["checkpoint_sha256"],
        "source_factorized_checkpoint_sha256":i8meta["source_factorized_checkpoint_sha256"],
        "source_l16_checkpoint_sha256":i8meta["source_l16_checkpoint_sha256"],
        "decision":decision,
    }
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if a.output:
        path=Path(a.output)
        if path.exists():
            raise SystemExit(f"refusing to overwrite resource evaluation: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if decision["status"]=="PACKED_INT4_RESOURCE_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
