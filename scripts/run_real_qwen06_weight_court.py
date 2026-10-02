from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

from nolane_personal.factorized_boundary import (
    FactorizedBoundaryConfig,
    analytical_factorized_boundary_parameters,
    dense_boundary_parameters,
    factorize_standalone_boundary,
)
from nolane_personal.quantized_boundary import (
    QuantizedBoundaryConfig,
    analytical_factorized_storage_bytes,
    analytical_quantized_storage_bytes,
    quantize_factorized_boundary,
)
from nolane_personal.real_qwen06_court import RealQwen06Evidence, decide_real_qwen06_court
from nolane_personal.standalone_model import StandaloneBoundaryModule, StandaloneNolaneConfig

ROOT=Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deterministic_rows(torch,vocab_size,count):
    count=min(int(count),int(vocab_size))
    rows=(torch.arange(count,dtype=torch.long)*int(vocab_size))//count
    return rows


def relative_error(torch,left,right):
    denom=left.detach().float().norm().clamp_min(1e-12)
    return float(((left.detach().float()-right.detach().float()).norm()/denom).cpu())


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B"))
    parser.add_argument("--lock",default=str(ROOT/"model.lock.json"))
    parser.add_argument("--sample-rows",type=int,default=512)
    parser.add_argument("--rank",type=int,default=128)
    parser.add_argument("--output",default="runtime-data/l20-real-qwen06-court.json")
    args=parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install Qwen court dependencies first") from exc

    lock=load_json(args.lock)
    upstream=lock["upstream"]
    expected_params=int(upstream["parameters_bf16"])
    revision=str(upstream["revision"])
    marker=Path(args.model)/".nolane-model-revision"
    if not marker.exists():
        raise SystemExit("pinned model revision marker missing")
    resolved=marker.read_text(encoding="utf-8").strip()

    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        local_files_only=True,
    )
    model.eval()

    total_params=sum(parameter.numel() for parameter in model.parameters())
    config=model.config
    vocab=int(config.vocab_size)
    hidden=int(config.hidden_size)
    layers=len(model.model.layers)
    embed=model.model.embed_tokens
    lm_head=model.lm_head
    tied=bool(embed.weight.data_ptr()==lm_head.weight.data_ptr())

    rendered="Xin chào. Hello."
    ids=tokenizer(rendered,return_tensors="pt")["input_ids"][:,:8]
    with torch.inference_mode():
        forward=model(input_ids=ids,use_cache=False)
    logits=forward.logits
    forward_finite=bool(torch.isfinite(logits).all())
    top_token=int(torch.argmax(logits[0,-1]).item())

    rows=deterministic_rows(torch,vocab,args.sample_rows)
    input_rows=embed.weight.detach().cpu().index_select(0,rows).float().clone()
    output_rows=(
        input_rows.clone()
        if tied
        else lm_head.weight.detach().cpu().index_select(0,rows).float().clone()
    )
    norm_weight=model.model.norm.weight.detach().cpu().float().clone()

    source_boundary_bytes=embed.weight.numel()*embed.weight.element_size()
    source_boundary_bytes+=model.model.norm.weight.numel()*model.model.norm.weight.element_size()
    if not tied:
        source_boundary_bytes+=lm_head.weight.numel()*lm_head.weight.element_size()

    del forward,logits,model,embed,lm_head
    gc.collect()

    sampled_vocab=int(rows.numel())
    rank=min(int(args.rank),sampled_vocab,hidden)
    sample=StandaloneBoundaryModule(
        StandaloneNolaneConfig(
            vocab_size=sampled_vocab,
            hidden_size=hidden,
            rms_norm_eps=float(getattr(config,"rms_norm_eps",1e-6)),
            tie_word_embeddings=tied,
            padding_idx=None,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
            boundary_dtype="float32",
        )
    )
    with torch.no_grad():
        sample.embed_tokens.weight.copy_(input_rows)
        sample.final_norm.weight.copy_(norm_weight)
        if not tied:
            sample.lm_head.weight.copy_(output_rows)

    factorized,factor_receipt=factorize_standalone_boundary(sample,rank=rank,seed=20261002)
    quantized,quant_receipt=quantize_factorized_boundary(factorized,logit_chunk_size=128)

    torch.manual_seed(20261002)
    probe_hidden=torch.randn(1,4,hidden)
    with torch.inference_mode():
        float_logits=factorized.logits(probe_hidden)
        int8_logits=quantized.logits(probe_hidden)
    logit_error=relative_error(torch,float_logits,int8_logits)

    full_factor_cfg=FactorizedBoundaryConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        rank=rank,
        rms_norm_eps=float(getattr(config,"rms_norm_eps",1e-6)),
        tie_word_embeddings=tied,
    )
    dense_params=dense_boundary_parameters(full_factor_cfg)
    fact_params=analytical_factorized_boundary_parameters(full_factor_cfg)
    qcfg=QuantizedBoundaryConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        rank=rank,
        rms_norm_eps=float(getattr(config,"rms_norm_eps",1e-6)),
        tie_word_embeddings=tied,
    )
    fp32_bytes=analytical_factorized_storage_bytes(qcfg,source_element_size=4)
    bf16_bytes=analytical_factorized_storage_bytes(qcfg,source_element_size=2)
    qbytes=analytical_quantized_storage_bytes(qcfg)

    evidence=RealQwen06Evidence(
        repo_id=str(upstream["repo_id"]),
        requested_revision=revision,
        resolved_revision=resolved,
        model_type=str(config.model_type),
        parameter_count=int(total_params),
        expected_parameter_count=expected_params,
        vocab_size=vocab,
        hidden_size=hidden,
        decoder_layers=layers,
        tie_word_embeddings=tied,
        forward_sequence_tokens=int(ids.shape[1]),
        forward_logits_finite=forward_finite,
        forward_top_token_id=top_token,
        sampled_vocab_rows=sampled_vocab,
        sampled_rank=rank,
        sample_input_factorization_error=float(factor_receipt["input_relative_frobenius_error"]),
        sample_output_factorization_error=float(factor_receipt["output_relative_frobenius_error"]),
        sample_input_quantization_error=max(
            float(quant_receipt["input_code_relative_error"]),
            float(quant_receipt["input_basis_relative_error"]),
        ),
        sample_output_quantization_error=max(
            float(quant_receipt["output_code_relative_error"]),
            float(quant_receipt["output_basis_relative_error"]),
        ),
        sample_logit_relative_error=logit_error,
        full_dense_boundary_parameters=int(dense_params),
        full_factorized_boundary_parameters=int(fact_params),
        full_factorized_parameter_ratio=float(fact_params/max(1,dense_params)),
        full_factorized_fp32_bytes=int(fp32_bytes),
        full_factorized_bf16_bytes=int(bf16_bytes),
        full_quantized_bytes=int(qbytes),
        full_quantized_ratio_vs_fp32=float(qbytes/max(1,fp32_bytes)),
        full_quantized_ratio_vs_bf16=float(qbytes/max(1,bf16_bytes)),
        source_boundary_storage_bytes=int(source_boundary_bytes),
        runtime_requires_qwen_for_court=True,
    )
    decision=decide_real_qwen06_court(evidence)
    result={
        "schema":"NOLANE-L20-REAL-QWEN06-WEIGHT-COURT-V1",
        "authority":"MECHANICAL_EVIDENCE_ONLY_NO_QUALITY_PROMOTION",
        "decision":decision,
    }
    rendered=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(rendered+"\n",encoding="utf-8")
    print(rendered)
    return 0 if decision["status"]=="REAL_QWEN06_WEIGHT_COURT_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
