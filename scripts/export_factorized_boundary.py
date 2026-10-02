from __future__ import annotations

import argparse,json
from pathlib import Path

from nolane_personal.factorized_artifact import export_factorized_from_l16
from nolane_personal.latent import LatentStore

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--l16",default="runtime-data/l16-standalone/standalone-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--output-dir",default="runtime-data/l17-factorized-boundary")
    p.add_argument("--rank",type=int,default=128)
    p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--dataset-fingerprint",default=None)
    a=p.parse_args()
    latent=LatentStore(a.latent).load()
    if latent is None: raise SystemExit(f"persistent latent not found: {a.latent}")
    _model,manifest,receipt=export_factorized_from_l16(
        a.l16,a.output_dir,latent=latent.values,rank=a.rank,device=a.device,seed=a.seed,
        expected_dataset_fingerprint=a.dataset_fingerprint,
    )
    print(json.dumps({"factorization":receipt,"artifact":manifest},indent=2,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
