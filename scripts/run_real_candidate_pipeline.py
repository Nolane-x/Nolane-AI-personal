from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nolane_personal.approved_evidence import resolve_approved_evidence_pack
from nolane_personal.evidence_chain import (
    EvidenceStageError,
    REAL_CANDIDATE_STAGE_ORDER,
    evidence_chain_contract_sha256,
    run_stage,
    sha256_file,
    verify_complete_stage_order,
    write_receipt,
)
from nolane_personal.real_candidate_readiness import assess_real_candidate_readiness


ROOT=Path(__file__).resolve().parents[1]


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-pack",default=None)
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

    approved_manifest=None
    if a.evidence_pack:
        approved_manifest,dataset_path,protocol_path=resolve_approved_evidence_pack(a.evidence_pack)
        a.dataset=str(dataset_path)
        a.protocol=str(protocol_path)

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
        "approved_evidence_manifest_sha256":(
            approved_manifest.get("manifest_sha256") if approved_manifest else None
        ),
        "approved_evidence_authority":(
            approved_manifest.get("authority") if approved_manifest else None
        ),
        "evidence_quality_status":readiness.evidence.get("evidence_quality_status"),
        "evidence_quality_court_sha256":readiness.evidence.get("evidence_quality_court_sha256"),
        "approved_evidence_quality_court_sha256":(
            approved_manifest.get("quality_court_sha256") if approved_manifest else None
        ),
        "stage_contract_sha256":evidence_chain_contract_sha256(),
        "stage_count_expected":len(REAL_CANDIDATE_STAGE_ORDER),
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

    def execute_stage(*,name,command,expected):
        try:
            run_stage(
                name=name,
                command=command,
                expected=expected,
                receipt=receipt,
                receipt_path=receipt_path,
                root=ROOT,
            )
        except EvidenceStageError as exc:
            raise SystemExit(exc.exit_code or 3) from exc

    execute_stage(
        name="freeze_l15_spec",
        command=[py,"scripts/freeze_native_boundary_spec.py",
                 "--model-lock",a.model_lock,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--output",str(spec)],
        expected=[spec],
    )
    execute_stage(
        name="train_l15_native",
        command=[py,"scripts/train_native_boundary.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--spec",str(spec),
                 "--output-dir",str(l15_dir),"--device",a.device],
        expected=[l15_ckpt,l15_dir/"native-nolane-boundary-manifest.json"],
    )
    execute_stage(
        name="evaluate_l15_quality",
        command=[py,"scripts/evaluate_native_boundary.py",
                 "--model-lock",a.model_lock,"--model",a.model,
                 "--native-boundary",str(l15_ckpt),"--l14-anchor",a.l14_anchor,
                 "--latent",a.latent,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--anchor",a.anchor,"--device",a.device,"--output",str(l15_quality)],
        expected=[l15_quality],
    )
    execute_stage(
        name="benchmark_l15_resources",
        command=[py,"scripts/benchmark_native_boundary_resources.py",
                 "--model-lock",a.model_lock,"--model",a.model,
                 "--native-boundary",str(l15_ckpt),"--l14-anchor",a.l14_anchor,
                 "--latent",a.latent,"--prompts",a.anchor,"--device",a.device,
                 "--output",str(l15_resources)],
        expected=[l15_resources],
    )
    execute_stage(
        name="promote_l15",
        command=[py,"scripts/decide_native_boundary_promotion.py",
                 "--quality",str(l15_quality),"--resources",str(l15_resources),
                 "--output",str(l15_promotion)],
        expected=[l15_promotion],
    )

    execute_stage(
        name="export_l16_standalone",
        command=[py,"scripts/export_standalone_nolane.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--latent",a.latent,"--output-dir",str(l16_dir),"--device",a.device,
                 "--dataset-fingerprint",str(protocol_sha)],
        expected=[l16_ckpt,l16_manifest],
    )
    execute_stage(
        name="evaluate_l16_parity",
        command=[py,"scripts/evaluate_standalone_parity.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--standalone",str(l16_ckpt),"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--device",a.device,
                 "--output",str(l16_parity)],
        expected=[l16_parity],
    )
    execute_stage(
        name="benchmark_l16_resources",
        command=[py,"scripts/benchmark_standalone_resources.py",
                 "--model-lock",a.model_lock,"--model",a.model,"--l15",str(l15_ckpt),
                 "--standalone",str(l16_ckpt),"--latent",a.latent,
                 "--prompts",a.anchor,"--device",a.device,"--output",str(l16_resources)],
        expected=[l16_resources],
    )
    execute_stage(
        name="promote_l16",
        command=[py,"scripts/decide_standalone_promotion.py",
                 "--parity",str(l16_parity),"--resources",str(l16_resources),
                 "--output",str(l16_promotion)],
        expected=[l16_promotion],
    )

    execute_stage(
        name="search_l18_rank_frontier",
        command=[py,"scripts/search_rank_frontier.py",
                 "--l16",str(l16_ckpt),"--latent",a.latent,"--dataset",a.dataset,
                 "--protocol",a.protocol,"--tokenizer",a.model,
                 "--output-dir",str(l18_dir),"--ranks",a.ranks,"--device",a.device],
        expected=[l18_ckpt,l18_frontier],
    )
    execute_stage(
        name="evaluate_l18_quality",
        command=[py,"scripts/evaluate_factorized_boundary.py",
                 "--l16",str(l16_ckpt),"--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--dataset",a.dataset,"--protocol",a.protocol,"--anchor",a.anchor,
                 "--tokenizer",a.model,"--device",a.device,"--output",str(l18_quality)],
        expected=[l18_quality],
    )
    execute_stage(
        name="benchmark_l18_resources",
        command=[py,"scripts/benchmark_factorized_resources.py",
                 "--l16",str(l16_ckpt),"--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--prompts",a.anchor,"--tokenizer",a.model,"--device",a.device,
                 "--output",str(l18_resources)],
        expected=[l18_resources],
    )
    execute_stage(
        name="promote_l18",
        command=[py,"scripts/decide_rank_frontier_promotion.py",
                 "--frontier",str(l18_frontier),"--quality",str(l18_quality),
                 "--resources",str(l18_resources),"--output",str(l18_promotion)],
        expected=[l18_promotion],
    )

    l16_meta=json.loads(l16_manifest.read_text(encoding="utf-8"))
    l16_sha=str(l16_meta["checkpoint_sha256"])
    execute_stage(
        name="export_l19_quantized",
        command=[py,"scripts/export_quantized_boundary.py",
                 "--factorized",str(l18_ckpt),"--latent",a.latent,
                 "--output-dir",str(l19_dir),"--device",a.device,
                 "--source-l16-sha",l16_sha,"--dataset-fingerprint",str(protocol_sha)],
        expected=[l19_ckpt,l19_dir/"quantized-nolane-manifest.json"],
    )
    execute_stage(
        name="evaluate_l19_quality",
        command=[py,"scripts/evaluate_quantized_boundary.py",
                 "--factorized",str(l18_ckpt),"--quantized",str(l19_ckpt),
                 "--latent",a.latent,"--dataset",a.dataset,"--protocol",a.protocol,
                 "--anchor",a.anchor,"--tokenizer",a.model,"--device",a.device,
                 "--output",str(l19_quality)],
        expected=[l19_quality],
    )
    execute_stage(
        name="benchmark_l19_resources",
        command=[py,"scripts/benchmark_quantized_resources.py",
                 "--factorized",str(l18_ckpt),"--quantized",str(l19_ckpt),
                 "--latent",a.latent,"--prompts",a.anchor,"--tokenizer",a.model,
                 "--device",a.device,"--output",str(l19_resources)],
        expected=[l19_resources],
    )
    execute_stage(
        name="promote_l19",
        command=[py,"scripts/decide_quantized_promotion.py",
                 "--quality",str(l19_quality),"--resources",str(l19_resources),
                 "--output",str(l19_promotion)],
        expected=[l19_promotion],
    )

    verify_complete_stage_order(receipt)
    receipt["status"]="REAL_CANDIDATE_EVIDENCE_CHAIN_PASS"
    receipt["blocked_stage"]=None
    receipt["stage_count_observed"]=len(receipt["stages"])
    receipt["final_quantized_checkpoint_sha256"]=sha256_file(l19_ckpt)
    receipt["final_promotion_sha256"]=sha256_file(l19_promotion)
    write_receipt(receipt_path,receipt)
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
