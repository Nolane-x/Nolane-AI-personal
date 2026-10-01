from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nolane_personal.native_spec import build_native_boundary_spec
from nolane_personal.personal_protocol import load_protocol, verify_personalization_protocol
from nolane_personal.surgery_candidate import load_model_lock, model_lock_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lock", default=str(ROOT / "model.lock.json"))
    parser.add_argument("--dataset", default="runtime-data/personalization.jsonl")
    parser.add_argument("--protocol", default="runtime-data/personalization-protocol-v1.json")
    parser.add_argument("--output", default="runtime-data/l15-native-boundary-spec.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    protocol = load_protocol(args.protocol)
    verify_personalization_protocol(
        protocol,
        dataset_sha256=sha256_file(dataset_path),
    )
    fingerprint = model_lock_fingerprint(load_model_lock(args.model_lock))
    spec = build_native_boundary_spec(
        base_model_fingerprint=fingerprint,
        dataset_fingerprint=protocol["protocol_sha256"],
    )

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen native-boundary spec: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
