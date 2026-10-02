from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .deep_recurrent_cortex import DeepRecurrentCortexConfig,DeepRecurrentStateSpaceCortex
from .factorized_boundary import FactorizedBoundaryConfig,FactorizedBoundaryModule,factorize_standalone_boundary
from .standalone_artifact import load_standalone_model
from .standalone_model import StandaloneNolaneLM
from .store import canonical_json,payload_digest
from .surgery import module_parameter_digest

SCHEMA="NOLANE-L17-FACTORIZED-LANGUAGE-BOUNDARY-V1"


def save_factorized_artifact(output_dir,model,factorization_receipt,*,source_meta,dataset_fingerprint=None,training_receipt=None):
    torch=model.cortex.torch
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=output_dir/"factorized-nolane.pt"
    payload={
        "schema":SCHEMA,
        "authority":"FACTORIZED_CANDIDATE_UNPROMOTED",
        "source_l16_checkpoint_sha256":source_meta["checkpoint_sha256"],
        "source_l15_checkpoint_sha256":source_meta.get("source_l15_checkpoint_sha256"),
        "source_base_model_fingerprint":source_meta.get("source_base_model_fingerprint"),
        "dataset_fingerprint":dataset_fingerprint or source_meta.get("dataset_fingerprint"),
        "factorized_config":asdict(model.boundary.config),
        "cortex_config":asdict(model.cortex.config),
        "boundary_state":model.boundary.state_dict(),
        "cortex_state":model.cortex.module.state_dict(),
        "factorization_receipt":factorization_receipt,
        "training_receipt":training_receipt,
        "carry_recurrent_state":model.carry_recurrent_state,
        "runtime_requires_qwen_model":False,
        "runtime_requires_transformers":False,
    }
    torch.save(payload,checkpoint)
    manifest={
        "schema":SCHEMA,
        "authority":payload["authority"],
        "source_l16_checkpoint_sha256":payload["source_l16_checkpoint_sha256"],
        "dataset_fingerprint":payload["dataset_fingerprint"],
        "factorized_config":payload["factorized_config"],
        "factorization_receipt":factorization_receipt,
        "training_receipt":training_receipt,
        "boundary_parameters":model.boundary.parameter_count(),
        "cortex_parameters":model.cortex.parameter_count(),
        "total_parameters":model.total_parameter_count(),
        "checkpoint_bytes":checkpoint.stat().st_size,
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "boundary_state_digest":module_parameter_digest(model.boundary.module),
        "cortex_state_digest":module_parameter_digest(model.cortex.module),
        "runtime_requires_qwen_model":False,
        "runtime_requires_transformers":False,
    }
    manifest["artifact_id"]=payload_digest(manifest)
    (output_dir/"factorized-nolane-manifest.json").write_text(canonical_json(manifest)+"\n",encoding="utf-8")
    return manifest


def export_factorized_from_l16(l16_checkpoint,output_dir,*,latent,rank:int=128,device:str="cpu",seed:int=0,expected_dataset_fingerprint=None):
    source,meta=load_standalone_model(
        l16_checkpoint,latent,device=device,expected_dataset_fingerprint=expected_dataset_fingerprint
    )
    boundary,receipt=factorize_standalone_boundary(source.boundary,rank=rank,seed=seed)
    cortex=DeepRecurrentStateSpaceCortex(
        source.cortex.hidden_size,DeepRecurrentCortexConfig(**asdict(source.cortex.config)),seed=0
    )
    cortex.module.load_state_dict(source.cortex.module.state_dict())
    model=StandaloneNolaneLM(
        boundary,cortex,latent,carry_recurrent_state=source.carry_recurrent_state
    ).to(device).eval()
    manifest=save_factorized_artifact(
        output_dir,
        model,
        receipt,
        source_meta=meta,
        dataset_fingerprint=expected_dataset_fingerprint or meta.get("dataset_fingerprint"),
    )
    return model,manifest,receipt


def load_factorized_model(checkpoint_path,latent,*,device:str="cpu",expected_source_l16_checkpoint_sha256=None,expected_dataset_fingerprint=None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Factorized Nolane requires PyTorch") from exc
    path=Path(checkpoint_path)
    try:
        payload=torch.load(path,map_location=device,weights_only=True)
    except TypeError:
        payload=torch.load(path,map_location=device)
    if payload.get("schema")!=SCHEMA:
        raise ValueError("unsupported factorized Nolane schema")
    if payload.get("authority")!="FACTORIZED_CANDIDATE_UNPROMOTED":
        raise ValueError("factorized Nolane authority mismatch")
    if payload.get("runtime_requires_qwen_model") is not False or payload.get("runtime_requires_transformers") is not False:
        raise ValueError("factorized artifact declares forbidden runtime dependency")
    if expected_source_l16_checkpoint_sha256 and payload.get("source_l16_checkpoint_sha256")!=expected_source_l16_checkpoint_sha256:
        raise ValueError("factorized source L16 mismatch")
    if expected_dataset_fingerprint and payload.get("dataset_fingerprint")!=expected_dataset_fingerprint:
        raise ValueError("factorized dataset mismatch")
    config=FactorizedBoundaryConfig(**payload["factorized_config"])
    boundary=FactorizedBoundaryModule(config)
    boundary.load_state_dict(payload["boundary_state"])
    cortex=DeepRecurrentStateSpaceCortex(
        config.hidden_size,DeepRecurrentCortexConfig(**payload["cortex_config"]),seed=0
    )
    cortex.module.load_state_dict(payload["cortex_state"])
    model=StandaloneNolaneLM(
        boundary,cortex,latent,carry_recurrent_state=bool(payload.get("carry_recurrent_state",False))
    ).to(device).eval()
    meta={
        "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_l16_checkpoint_sha256":payload["source_l16_checkpoint_sha256"],
        "dataset_fingerprint":payload.get("dataset_fingerprint"),
        "factorization_receipt":payload["factorization_receipt"],
        "training_receipt":payload.get("training_receipt"),
        "boundary_state_digest":module_parameter_digest(boundary.module),
        "cortex_state_digest":module_parameter_digest(cortex.module),
        "runtime_requires_qwen_model":False,
        "runtime_requires_transformers":False,
    }
    return model,meta
