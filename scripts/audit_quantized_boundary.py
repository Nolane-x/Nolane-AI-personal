from __future__ import annotations

import json

from nolane_personal.quantized_boundary import (
    QuantizedBoundaryConfig,
    analytical_factorized_storage_bytes,
    analytical_quantized_storage_bytes,
)


def main() -> int:
    cfg=QuantizedBoundaryConfig(
        vocab_size=151936,
        hidden_size=1024,
        rank=128,
        tie_word_embeddings=True,
    )
    quantized=analytical_quantized_storage_bytes(cfg)
    fp32=analytical_factorized_storage_bytes(cfg,source_element_size=4)
    bf16=analytical_factorized_storage_bytes(cfg,source_element_size=2)
    result={
        "schema":"NOLANE-L19-QUANTIZED-STORAGE-AUDIT-V1",
        "vocab_size":cfg.vocab_size,
        "hidden_size":cfg.hidden_size,
        "rank":cfg.rank,
        "tied":cfg.tie_word_embeddings,
        "quantized_bytes":quantized,
        "fp32_factorized_bytes":fp32,
        "bf16_factorized_bytes":bf16,
        "ratio_vs_fp32":quantized/fp32,
        "ratio_vs_bf16":quantized/bf16,
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    if result["ratio_vs_fp32"]>=0.30:
        raise SystemExit("int8 boundary failed fp32 storage-ratio gate")
    if result["ratio_vs_bf16"]>=0.60:
        raise SystemExit("int8 boundary failed bf16 storage-ratio gate")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
