from __future__ import annotations

import json

from nolane_personal.packed_int4_boundary import (
    PackedInt4BoundaryConfig,
    analytical_int8_storage_bytes,
    analytical_packed_int4_storage_bytes,
)


def main() -> int:
    cfg=PackedInt4BoundaryConfig(
        vocab_size=151936,
        hidden_size=1024,
        rank=128,
        tie_word_embeddings=True,
    )
    int8=analytical_int8_storage_bytes(cfg)
    int4=analytical_packed_int4_storage_bytes(cfg)
    result={
        "schema":"NOLANE-L20-PACKED-INT4-STORAGE-AUDIT-V1",
        "vocab_size":cfg.vocab_size,
        "hidden_size":cfg.hidden_size,
        "rank":cfg.rank,
        "tied":cfg.tie_word_embeddings,
        "int8_bytes":int8,
        "packed_int4_bytes":int4,
        "ratio_vs_int8":int4/int8,
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    if result["ratio_vs_int8"]>=0.55:
        raise SystemExit("packed INT4 failed storage-ratio gate versus INT8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
