from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from nolane_personal.real_candidate_readiness import assess_real_candidate_readiness
from nolane_personal.store import canonical_json, payload_digest


ROOT=Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(path: Path, receipt: dict) -> None:
    body=dict(receipt)
    body.pop("receipt_sha256",None)
    receipt["receipt_sha256"]=payload_digest(body)
    path.write_text(canonical_json(receipt)+"\n",encoding="utf-8")


def stage_output_digest(paths: list[Path]) -> dict[str,str]:
    result={}
    for path in paths:
        if path.exists() and path.is_file():
            result[str(path)]=sha256_file(path)
    return result


def run_stage(*,name:str,command:list[str],expected:list[Path],receipt:dict,receipt_path:Path) -> None:
    print(f"\n[L21] {name}")
    print(" ".join(command))
    completed=subprocess.run(command,cwd=ROOT)
    row={
        "name":name,
        "exit_code":int(completed.returncode),
        "expected_outputs":[str(p) for p in expected],
        "output_sha256":stage_output_digest(expected),
    }
    missing=[str(p) for p in expected if not p.exists()]
    if missing:
        row["missing_outputs"]=missing
    receipt["stages"].append(row)
    if completed.returncode!=0 or missing:
        receipt["status"]="REAL_CANDIDATE_PIPELINE_BLOCKED"
        receipt["blocked_stage"]=name
        write_receipt(receipt_path,receipt)
        raise SystemExit(completed.returncode or 3)
    write_receipt(receipt_path,receipt)


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="runtime-data/personalization.jsonl")
    p.add_argument("--protocol",default="runtime-data/personalization-protocol-v1.json")
    p.add_argument("--anchor",default=str(ROOT/"research/personalization-general-anchor.jsonl"))
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--model-lock",default=str(ROOT/"model.lock.json"))
    p.add_argument("--model",default=str(ROOT/"models/Qwen3-0.6B"))
    p.add_argument("--l14-anchor",default="runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt")
    p.add_argument("--workspace",default="runtime-data/l21-real-candidate")
    p.add_argument("--ranks",default="256,192,128,96,64")
    p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()

    workspace=Path(a.workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(
            f"refusing to mix evidence in non-empty workspace: {workspace}; use a fresh directory"
        )
    workspace.mkdir(parents=True,exist_ok=True)
    receipt_path=workspace/"pipeline-receipt.json"

    readiness=assess_real_candidate_readiness(
        dataset=a.dataset,
        protocol=a.protocol,
        anchor=a.anchor,
        latent=a.latent,
        model_lock=a.model_lock,
        model_dir=a.model,
        l14_anchor=a.l14_anchor,
    )
    protocol_sha=readiness.evidence.get("protocol_sha256")
    receipt={
        "schema":"NOLANE-L21-REAL-CANDIDATE-PIPELINE-V1",
        "authority":"REAL_EVIDENCE_CHAIN_UNPROMOTED",
        "status":"REAL_CANDIDATE_INPUTS_READY" if readiness.status=="REAL_CANDIDATE_INPUTS_READY" else "REAL_CANDIDATE_PIPELINE_BLOCKED",
        "blocked_stage":None,
        "readiness":asdict(readiness),
        "dataset_sha256":readiness.evidence.get("dataset_sha256"),
        "protocol_sha256":protocol_sha,
        "model_revision":readiness.evidence.get("resolved_model_revision"),
        "stages":[],
    }
    if readiness.status!="REAL_CANDIDATE_INPUTS_READY":
        receipt["blocked_stage"]="readiness"
        write_receipt(receipt_path,receipt)
        print(json.dumps(receipt,indent=2,sort_keys=True))
        return 2

    if not a.execute:
        receipt["status"]="REAL_CANDIDATE_READY_NOT_EXECUTED"
        write_receipt(receipt_path,receipt)
        print(json.dumps(receipt,indent=2,sort_keys=True))
        return 0

    py=sys.executable
    spec=workspace/"l15-native-boundary-spec.json"
    l15_dir=workspace/"l15-native-boundary"
    l15_ckpt=l15_dir/"native-nolane-boundary.pt"
    l15_quality=workspace/"l15-quality.json"
    l15_resources=workspace/"l15-resources.json"
    l15_promotion=workspace/"l15-promotion.json"

    l16_dir=workspace/"l16-standalone"
    l16_ckpt=l16_dir/"standalone-nolane.pt"
    l16_manifest=l16_dir/"standalone-nolane-manifest.json"
    l16_parity=workspace/"l16-parity.json"
    l16_resources=workspace/"l16-resources.json"
    l16_promotion=workspace/"l16-promotion.json"

    l18_dir=workspace/"l18-rank-frontier"
    l18_ckpt=l18_dir/"factorized-nolane.pt"
    l18_frontier=l18_dir/"rank-frontier-receipt.json"
    l18_quality=workspace/"l18-quality.json"
    l18_resources=workspace/"l18-resources.json"
    l18_promotion=workspace/"l18-promotion.json"

    l19_dir=workspace/"l19-quantized"
    l19_ckpt=l19_dir/"quantized-nolane.pt"
    l19_quality=workspace/"l19-quality.json"
    l19_resources=workspace/"l19-resources.json"
    l19_promotion=workspace/"l19-promotion.json"

    receipt["status"]="REAL_CANDIDATE_PIPELINE_RUNNING"
    write_receipt(receipt_path,receipt)

    run_stage(
        name="freeze_l15_spec",
        command=[py,"scripts/freeze_native_boundary_spec.py",
                 "--model-lock",a.model_lock,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--output",str(spec)],
        expected=[spec],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="train_l15_native",
        command=[py,"scripts/train_native_boundary.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--spec",str(spec),
                 "--output-dir",str(l15_dir),"--device",a.device],
        expected=[l15_ckpt,l15_dir/"native-nolane-boundary-manifest.json"],
        receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="evaluate_l15_quality",
        command=[py,"scripts/evaluate_native_boundary.py",
                 "--model-lock",a.model_lock,"--model",a.model,
                 "--native-boundary",str(l15_ckpt),"--l14-anchor",a.l14_anchor,
                 "--latent",a.latent,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--anchor",a.anchor,"--device",a.device,"--output",str(l15_quality)],
        expected=[l15_quality],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="benchmark_l15_resources",
        command=[py,"scripts/benchmark_native_boundary_resources.py",
                 "--model-lock",a.model_lock,"--model",a.model,
                 "--native-boundary",str(l15_ckpt),"--l14-anchor",a.l14_anchor,
                 "--latent",a.latent,"--prompts",a.anchor,"--device",a.device,
                 "--output",str(l15_resources)],
        expected=[l15_resources],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="promote_l15",
        command=[py,"scripts/decide_native_boundary_promotion.py",
                 "--quality",str(l15_quality),"--resources",str(l15_resources),
                 "--output",str(l15_promotion)],
        expected=[l15_promotion],receipt=receipt,receipt_path=receipt_path,
    )

    run_stage(
        name="export_l16_standalone",
        command=[py,"scripts/export_standalone_nolane.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--latent",a.latent,"--output-dir",str(l16_dir),"--device",a.device,
                 "--dataset-fingerprint",str(protocol_sha)],
        expected=[l16_ckpt,l16_manifest],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="evaluate_l16_parity",
        command=[py,"scripts/evaluate_standalone_parity.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--standalone",str(l16_ckpt),"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--device",a.device,
                 "--output",str(l16_parity)],
        expected=[l16_parity],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="benchmark_l16_resources",
        command=[py,"scripts/benchmark_standalone_resources.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--standalone",str(l16_ckpt),"--latent",a.latent,
                 "--prompts",a.anchor,"--device",a.device,"--output",str(l16_resources)],
        expected=[l16_resources],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="promote_l16",
        command=[py,"scripts/decide_standalone_promotion.py",
                 "--parity",str(l16_parity),"--resources",str(l16_resources),
                 "--output",str(l16_promotion)],
        expected=[l16_promotion],receipt=receipt,receipt_path=receipt_path,
    )

    run_stage(
        name="search_l18_rank_frontier",
        command=[py,"scripts/search_rank_frontier.py",
                 "--l16",str(l16_ckpt),"--latent",a.latent,"--dataset",a.dataset,
                 "--protocol",a.protocol,"--tokenizer",a.model,
                 "--output-dir",str(l18_dir),"--ranks",a.ranks,"--device",a.device],
        expected=[l18_ckpt,l18_frontier],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="evaluate_l18_quality",
        command=[py,"scripts/evaluate_factorized_boundary.py",
                 "--l16",str(l16_ckpt),"--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--anchor",a.anchor,
                 "--tokenizer",a.model,"--device",a.device,"--output",str(l18_quality)],
        expected=[l18_quality],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="benchmark_l18_resources",
        command=[py,"scripts/benchmark_factorized_resources.py",
                 "--l16",str(l16_ckpt),"--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--prompts",a.anchor,"--tokenizer",a.model,"--device",a.device,
                 "--output",str(l18_resources)],
        expected=[l18_resources],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="promote_l18",
        command=[py,"scripts/decide_rank_frontier_promotion.py",
                 "--frontier",str(l18_frontier),"--quality",str(l18_quality),
                 "--resources",str(l18_resources),"--output",str(l18_promotion)],
        expected=[l18_promotion],receipt=receipt,receipt_path=receipt_path,
    )

    l16_meta=json.loads(l16_manifest.read_text(encoding="utf-8"))
    l16_sha=str(l16_meta["checkpoint_sha256"])
    run_stage(
        name="export_l19_quantized",
        command=[py,"scripts/export_quantized_boundary.py",
                 "--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--output-dir",str(l19_dir),"--device",a.device,
                 "--source-l16-sha",l16_sha,"--dataset-fingerprint",str(protocol_sha)],
        expected=[l19_ckpt,l19_dir/"quantized-nolane-manifest.json"],
        receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="evaluate_l19_quality",
        command=[py,"scripts/evaluate_quantized_boundary.py",
                 "--factorized",str(l18_ckpt),"--quantized",str(l19_ckpt),
                 "--latent",a.latent,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--anchor",a.anchor,"--tokenizer",a.model,"--device",a.device,
                 "--output",str(l19_quality)],
        expected=[l19_quality],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="benchmark_l19_resources",
        command=[py,"scripts/benchmark_quantized_resources.py",
                 "--factorized",str(l18_ckpt),"--quantized",str(l19_ckpt),
                 "--latent",a.latent,"--prompts",a.anchor,"--tokenizer",a.model,
                 "--device",a.device,"--output",str(l19_resources)],
        expected=[l19_resources],receipt=receipt,receipt_path=receipt_path,
    )
    run_stage(
        name="promote_l19",
        command=[py,"scripts/decide_quantized_promotion.py",
                 "--quality",str(l19_quality),"--resources",str(l19_resources),
                 "--output",str(l19_promotion)],
        expected=[l19_promotion],receipt=receipt,receipt_path=receipt_path,
    )

    receipt["status"]="REAL_CANDIDATE_EVIDENCE_CHAIN_PASS"
    receipt["blocked_stage"]=None
    receipt["final_quantized_checkpoint_sha256"]=sha256_file(l19_ckpt)
    receipt["final_promotion_sha256"]=sha256_file(l19_promotion)
    write_receipt(receipt_path,receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
