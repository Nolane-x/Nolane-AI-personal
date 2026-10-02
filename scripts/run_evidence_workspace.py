from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nolane_personal.evidence_workspace import verify_evidence_workspace


ROOT=Path(__file__).resolve().parents[1]


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument(
        "--spec",
        default="runtime-data/local-evidence-workspace/evidence-workspace.json",
    )
    p.add_argument(
        "--run-workspace",
        default="runtime-data/l21-real-candidate",
    )
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()

    spec=verify_evidence_workspace(a.spec)
    paths=spec["paths"]
    command=[
        sys.executable,
        str(ROOT/"scripts/run_real_candidate_pipeline.py"),
        "--evidence-pack",paths["evidence_pack_manifest"],
        "--anchor",paths["anchor"],
        "--latent",paths["latent"],
        "--model-lock",paths["model_lock"],
        "--model",paths["model_dir"],
        "--l14-anchor",paths["l14_anchor"],
        "--workspace",a.run_workspace,
        "--device",a.device,
    ]
    if a.execute:
        command.append("--execute")
    completed=subprocess.run(command,cwd=ROOT)
    return int(completed.returncode)


if __name__=="__main__":
    raise SystemExit(main())
