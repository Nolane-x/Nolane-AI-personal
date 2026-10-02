import hashlib
import json
import subprocess
import sys
from pathlib import Path

from nolane_personal.latent import LatentStore
from nolane_personal.personal_dataset import load_jsonl
from nolane_personal.personal_protocol import build_personalization_protocol


ROOT=Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, count: int, prefix: str):
    with path.open("w",encoding="utf-8") as fh:
        for i in range(count):
            fh.write(json.dumps({
                "prompt":f"{prefix}-private-prompt-{i}",
                "target":f"{prefix}-private-target-{i}",
                "language":"vi" if i%2==0 else "en",
            },ensure_ascii=False)+"\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path: Path):
    dataset=tmp_path/"data.jsonl"
    protocol=tmp_path/"protocol.json"
    anchor=tmp_path/"anchor.jsonl"
    latent=tmp_path/"latent.json"
    model=tmp_path/"model"
    lock=tmp_path/"model.lock.json"
    l14=tmp_path/"l14.pt"

    write_jsonl(dataset,7,"personal")
    examples=load_jsonl(dataset)
    frozen=build_personalization_protocol(examples,dataset_sha256=sha(dataset))
    protocol.write_text(json.dumps(frozen),encoding="utf-8")
    write_jsonl(anchor,4,"anchor")
    LatentStore(latent).initialize(
        identity_id="pipeline-test",
        checkpoint_sha256="core",
        latent_dim=32,
        protocol_sha256=None,
    )

    revision="real-revision"
    model.mkdir()
    (model/"config.json").write_text("{}",encoding="utf-8")
    (model/"model.safetensors").write_bytes(b"x")
    (model/".nolane-model-revision").write_text(revision,encoding="utf-8")
    lock.write_text(json.dumps({"upstream":{"revision":revision}}),encoding="utf-8")
    l14.write_bytes(b"l14")

    return dataset,protocol,anchor,latent,model,lock,l14


def command(tmp_path: Path, *, remove_dataset=False):
    dataset,protocol,anchor,latent,model,lock,l14=inputs(tmp_path)
    if remove_dataset:
        dataset.unlink()
    workspace=tmp_path/"workspace"
    cmd=[
        sys.executable,
        str(ROOT/"scripts/run_real_candidate_pipeline.py"),
        "--dataset",str(dataset),
        "--protocol",str(protocol),
        "--anchor",str(anchor),
        "--latent",str(latent),
        "--model-lock",str(lock),
        "--model",str(model),
        "--l14-anchor",str(l14),
        "--workspace",str(workspace),
    ]
    return cmd,workspace


def test_pipeline_check_only_writes_privacy_preserving_ready_receipt(tmp_path):
    cmd,workspace=command(tmp_path)
    completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    receipt_path=workspace/"pipeline-receipt.json"
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"]=="REAL_CANDIDATE_READY_NOT_EXECUTED"
    assert receipt["blocked_stage"] is None
    assert receipt["stages"]==[]
    rendered=receipt_path.read_text(encoding="utf-8")
    assert "private-prompt" not in rendered
    assert "private-target" not in rendered


def test_pipeline_blocks_at_readiness_before_any_training(tmp_path):
    cmd,workspace=command(tmp_path,remove_dataset=True)
    completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert completed.returncode==2
    receipt=json.loads((workspace/"pipeline-receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="REAL_CANDIDATE_PIPELINE_BLOCKED"
    assert receipt["blocked_stage"]=="readiness"
    assert receipt["stages"]==[]
    assert "personalization_dataset_missing" in receipt["readiness"]["reasons"]
