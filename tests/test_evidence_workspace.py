import json
import subprocess
import sys
from pathlib import Path

import pytest

from nolane_personal.approved_evidence import build_approved_evidence_pack
from nolane_personal.evidence_workspace import (
    prepare_evidence_workspace,
    verify_evidence_workspace,
)
from nolane_personal.latent import LatentStore


ROOT=Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8",
    )


def make_ready_inputs(tmp_path: Path):
    source=tmp_path/"approved-source.jsonl"
    write_jsonl(source,[
        {
            "prompt":f"PRIVATE-WORKSPACE-PROMPT-{i}",
            "target":f"PRIVATE-WORKSPACE-TARGET-{i}",
            "language":"vi" if i%2==0 else "en",
            "approved":True,
            "source_id":f"private-source-{i}",
        }
        for i in range(8)
    ])
    pack=build_approved_evidence_pack(source,tmp_path/"pack")

    anchor=tmp_path/"anchor.jsonl"
    write_jsonl(anchor,[
        {"prompt":f"anchor-{i}","target":f"answer-{i}","language":"vi"}
        for i in range(4)
    ])

    latent=tmp_path/"latent.json"
    LatentStore(latent).initialize(
        identity_id="workspace-test",
        checkpoint_sha256="core-checkpoint",
        latent_dim=32,
        protocol_sha256=None,
    )

    revision="pinned-revision"
    model=tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text(
        json.dumps({"model_type":"qwen3","hidden_size":1024}),
        encoding="utf-8",
    )
    (model/"model.safetensors").write_bytes(b"fake-model-weights")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")

    lock=tmp_path/"model.lock.json"
    lock.write_text(
        json.dumps({"upstream":{"revision":revision}}),
        encoding="utf-8",
    )
    l14=tmp_path/"l14.pt"
    l14.write_bytes(b"l14-candidate")
    return pack,anchor,latent,model,lock,l14


def test_workspace_binds_ready_inputs_without_copying_private_text(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    spec=verify_evidence_workspace(result["spec_path"])
    assert spec["status"]=="LOCAL_EVIDENCE_WORKSPACE_READY"
    assert spec["authority"]=="LOCAL_EXECUTION_WORKSPACE_UNPROMOTED"
    assert spec["readiness"]["status"]=="REAL_CANDIDATE_INPUTS_READY"
    assert spec["binding"]["evidence_manifest_sha256"]==pack.manifest["manifest_sha256"]
    assert spec["binding"]["requested_model_revision"]=="pinned-revision"
    assert spec["binding"]["resolved_model_revision"]=="pinned-revision"
    assert spec["binding"]["model_weight_files"]==[
        {"filename":"model.safetensors","bytes":len(b"fake-model-weights")}
    ]
    rendered=result["spec_path"].read_text(encoding="utf-8")
    assert "PRIVATE-WORKSPACE-PROMPT" not in rendered
    assert "PRIVATE-WORKSPACE-TARGET" not in rendered
    assert "private-source-" not in rendered


@pytest.mark.parametrize(
    "target,expected",
    [
        ("anchor","anchor digest mismatch"),
        ("l14","l14_anchor digest mismatch"),
        ("lock","model_lock digest mismatch"),
    ],
)
def test_workspace_detects_bound_file_tamper(tmp_path,target,expected):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    mapping={"anchor":anchor,"l14":l14,"lock":lock}
    path=mapping[target]
    if path.suffix==".jsonl":
        path.write_text(path.read_text(encoding="utf-8")+"{}\n",encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match=expected):
        verify_evidence_workspace(result["spec_path"])


def test_workspace_detects_latent_tamper(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    payload=json.loads(latent.read_text(encoding="utf-8"))
    payload["values"][0]=123.0
    latent.write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="latent digest mismatch"):
        verify_evidence_workspace(result["spec_path"])


def test_workspace_detects_model_weight_inventory_change(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    (model/"model.safetensors").write_bytes(b"changed-length-model-weights")
    with pytest.raises(ValueError,match="model weight inventory mismatch"):
        verify_evidence_workspace(result["spec_path"])


def test_workspace_detects_model_config_or_revision_change(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    (model/"config.json").write_text('{"tampered":true}',encoding="utf-8")
    with pytest.raises(ValueError,match="model config digest mismatch"):
        verify_evidence_workspace(result["spec_path"])

    # Restore config from a fresh fixture and test marker independently.
    pack2,anchor2,latent2,model2,lock2,l142=make_ready_inputs(tmp_path/"second")
    result2=prepare_evidence_workspace(
        evidence_pack_manifest=pack2.manifest_path,
        anchor=anchor2,
        latent=latent2,
        model_lock=lock2,
        model_dir=model2,
        l14_anchor=l142,
        output_dir=tmp_path/"workspace-second",
    )
    (model2/".nolane-model-revision").write_text("different",encoding="utf-8")
    with pytest.raises(ValueError,match="model resolved revision mismatch"):
        verify_evidence_workspace(result2["spec_path"])


def test_workspace_refuses_non_empty_output(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    output=tmp_path/"workspace"
    output.mkdir()
    (output/"existing.txt").write_text("do not mix",encoding="utf-8")
    with pytest.raises(FileExistsError,match="refusing to overwrite"):
        prepare_evidence_workspace(
            evidence_pack_manifest=pack.manifest_path,
            anchor=anchor,
            latent=latent,
            model_lock=lock,
            model_dir=model,
            l14_anchor=l14,
            output_dir=output,
        )


def test_workspace_runner_drives_l21_check_only(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    run_workspace=tmp_path/"l21-run"
    completed=subprocess.run(
        [
            sys.executable,
            str(ROOT/"scripts/run_evidence_workspace.py"),
            "--spec",str(result["spec_path"]),
            "--run-workspace",str(run_workspace),
            "--device","cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode==0,completed.stderr
    receipt=json.loads(
        (run_workspace/"pipeline-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"]=="REAL_CANDIDATE_READY_NOT_EXECUTED"
    assert receipt["approved_evidence_manifest_sha256"]==pack.manifest["manifest_sha256"]
    rendered=(run_workspace/"pipeline-receipt.json").read_text(encoding="utf-8")
    assert "PRIVATE-WORKSPACE-PROMPT" not in rendered
    assert "PRIVATE-WORKSPACE-TARGET" not in rendered


def test_workspace_verification_detects_spec_tamper(tmp_path):
    pack,anchor,latent,model,lock,l14=make_ready_inputs(tmp_path)
    result=prepare_evidence_workspace(
        evidence_pack_manifest=pack.manifest_path,
        anchor=anchor,
        latent=latent,
        model_lock=lock,
        model_dir=model,
        l14_anchor=l14,
        output_dir=tmp_path/"workspace",
    )
    payload=json.loads(result["spec_path"].read_text(encoding="utf-8"))
    payload["status"]="LOCAL_EVIDENCE_WORKSPACE_BLOCKED"
    result["spec_path"].write_text(json.dumps(payload),encoding="utf-8")
    with pytest.raises(ValueError,match="workspace digest mismatch"):
        verify_evidence_workspace(result["spec_path"])
