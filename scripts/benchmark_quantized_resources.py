from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from nolane_personal.factorized_artifact import load_factorized_model
from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.qwen import SYSTEM_PROMPT
from nolane_personal.quantized_artifact import load_quantized_model
from nolane_personal.quantized_evaluation import QuantizedResourceEvidence,decide_quantized_resources


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


def storage_bytes(module):
    return sum(t.numel()*t.element_size() for t in module.state_dict().values())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--factorized",default="runtime-data/l18-rank-frontier/factorized-nolane.pt")
    p.add_argument("--quantized",default="runtime-data/l19-quantized/quantized-nolane.pt")
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
    source,smeta=load_factorized_model(a.factorized,latent.values,device=a.device)
    candidate,qmeta=load_quantized_model(
        a.quantized,
        latent.values,
        device=a.device,
        expected_source_factorized_checkpoint_sha256=smeta["checkpoint_sha256"],
        expected_source_l16_checkpoint_sha256=smeta["source_l16_checkpoint_sha256"],
    )
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    samples=[
        {k:v.to(a.device) for k,v in dict(encode_prompt(tok,e.prompt)).items()}
        for e in load_jsonl(a.prompts)
    ]
    source_times=[]; quant_times=[]; rates=[]
    with torch.no_grad():
        for sample in samples:
            source_times.append(timed(lambda:source.forward(input_ids=sample["input_ids"],state=None)))
            quant_times.append(timed(lambda:candidate.forward(input_ids=sample["input_ids"],state=None)))
            start=time.perf_counter()
            out=candidate.generate(
                input_ids=sample["input_ids"],
                max_new_tokens=a.generation_tokens,
                do_sample=False,
                eos_token_id=None,
            )
            rates.append(
                max(1,out.shape[1]-sample["input_ids"].shape[1])
                / max(time.perf_counter()-start,1e-9)
            )

    source_median=statistics.median(source_times)
    quant_median=statistics.median(quant_times)
    source_boundary_bytes=storage_bytes(source.boundary.module)
    quant_boundary_bytes=candidate.boundary.storage_bytes()
    ev=QuantizedResourceEvidence(
        prompts=len(samples),
        source_median_ms=source_median,
        quantized_median_ms=quant_median,
        latency_ratio_vs_source=quant_median/max(source_median,1e-9),
        source_checkpoint_bytes=Path(a.factorized).stat().st_size,
        quantized_checkpoint_bytes=Path(a.quantized).stat().st_size,
        checkpoint_ratio=Path(a.quantized).stat().st_size/max(1,Path(a.factorized).stat().st_size),
        source_boundary_storage_bytes=source_boundary_bytes,
        quantized_boundary_storage_bytes=quant_boundary_bytes,
        boundary_storage_ratio=quant_boundary_bytes/max(1,source_boundary_bytes),
        tokens_per_second=statistics.median(rates),
    )
    decision=decide_quantized_resources(ev)
    result={
        "schema":"NOLANE-L19-QUANTIZED-RESOURCE-EVAL-V1",
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
            raise SystemExit(f"refusing to overwrite resource evaluation: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if decision["status"]=="QUANTIZED_FACTOR_RESOURCE_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
