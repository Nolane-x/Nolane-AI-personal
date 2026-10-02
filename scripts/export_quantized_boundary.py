from __future__ import annotations

import argparse
import json

from nolane_personal.latent import LatentStore
from nolane_personal.quantized_artifact import quantize_factorized_artifact


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--factorized",default="runtime-data/l18-rank-frontier/factorized-nolane.pt")
    p.add_argument("--latent",default="runtime-data/living-core-shadow/latent.json")
    p.add_argument("--output-dir",default="runtime-data/l19-quantized")
    p.add_argument("--device",default="cpu")
    p.add_argument("--logit-chunk-size",type=int,default=4096)
    p.add_argument("--source-l16-sha",default=None)
    p.add_argument("--dataset-fingerprint",default=None)
    a=p.parse_args()
    latent=LatentStore(a.latent).load()
    if latent is None:
        raise SystemExit(f"persistent latent not found: {a.latent}")
    _model,manifest,receipt=quantize_factorized_artifact(
        a.factorized,
        a.output_dir,
        latent=latent.values,
        device=a.device,
        logit_chunk_size=a.logit_chunk_size,
        expected_source_l16_checkpoint_sha256=a.source_l16_sha,
        expected_dataset_fingerprint=a.dataset_fingerprint,
    )
    print(json.dumps({"quantization":receipt,"artifact":manifest},indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
