# L19 Quantized Factor Runtime

## Purpose

L18 chooses the smallest low-rank language boundary justified by development evidence.

L19 compresses the selected low-rank factors themselves for inference.

The persistent boundary state becomes:

```text
token codes: int8 + per-row scale
basis:       int8 + per-row scale
RMSNorm:     float
```

The recurrent Nolane cortex is unchanged.

## Quantization

L19 uses symmetric row-wise int8 quantization.

For every row `x`:

```text
scale = max(abs(x)) / 127
q     = round(x / scale), clipped to [-127, 127]
x_hat = q * scale
```

Zero rows are handled safely by a minimum positive scale.

Input and output factors are quantized independently when embeddings are untied. When the language boundary is tied, the same quantized factors are reused for embedding and logits.

## Runtime memory behavior

The int8 code matrix remains int8 in persistent runtime state.

Embedding only dequantizes the token rows requested by the current input.

For output logits, the vocabulary code matrix is processed in chunks:

```text
normalized hidden
      |
      v
dequantized small rank x hidden basis
      |
      v
rank hidden
      |
      +--> dequant vocab chunk 0 -> logits chunk 0
      +--> dequant vocab chunk 1 -> logits chunk 1
      +--> ...
```

This avoids reconstructing a full float `vocab x rank` code matrix as persistent model state.

Default logit chunk size is 4096 rows.

## Byte-level storage court

The storage court counts actual tensor bytes rather than parameter names.

For a representative tied boundary:

- vocabulary: 151,936
- hidden size: 1,024
- rank: 128

the analytical L19 gate requires:

- int8 storage <30% of the equivalent FP32 factorized boundary;
- int8 storage <60% of the equivalent BF16/FP16 factorized boundary.

The exact audit runs in CI:

```bash
python scripts/audit_quantized_boundary.py
```

## Export

Quantization consumes an already-selected factorized checkpoint.

```bash
python scripts/export_quantized_boundary.py \
  --factorized runtime-data/l18-rank-frontier/factorized-nolane.pt \
  --output-dir runtime-data/l19-quantized
```

The artifact binds:

- exact source factorized checkpoint SHA-256;
- exact source L16 checkpoint SHA-256;
- dataset/protocol fingerprint;
- rank and token-boundary configuration;
- quantization errors;
- boundary state digest;
- cortex state digest.

It declares no Qwen model-object or Transformers-model runtime dependency.

## Held-out quality court

```bash
python scripts/evaluate_quantized_boundary.py
```

The default quality court requires:

- frozen held-out personal evidence;
- frozen Vietnamese/English anchor evidence;
- test NLL regression versus the factorized source <=0.02;
- anchor NLL regression <=0.03;
- greedy-token agreement >=0.98;
- prompt full-scan == incremental recurrent scan;
- selected rank unchanged;
- cortex digest unchanged;
- no Qwen/Transformers model dependency in the quantized runtime.

## Resource court

```bash
python scripts/benchmark_quantized_resources.py
```

The resource court measures the exact factorized and quantized checkpoints on the target device.

It checks:

- checkpoint compression;
- boundary tensor-byte compression;
- forward latency ratio;
- generation throughput.

The current pure-PyTorch path uses chunked dequantization. It is intentionally not described as a native int8 GEMM kernel. A candidate that saves storage but becomes too slow fails the resource court.

## Promotion

```bash
python scripts/decide_quantized_promotion.py \
  --quality runtime-data/l19-quality.json \
  --resources runtime-data/l19-resources.json
```

Promotion requires quality and resource evidence to reference:

- the same quantized checkpoint;
- the same factorized source checkpoint;
- the same L16 source checkpoint.

## Neural court evidence

CI verifies:

- zero-row-safe row-wise int8 quantization;
- realistic FP32 and BF16 storage ratios;
- tied factor reuse;
- untied output-factor handling;
- embedding closeness versus float factorized weights;
- logit closeness versus float factorized weights;
- prompt scan equivalence;
- generation compatibility;
- inference-only contract;
- exact artifact roundtrip;
- exact cortex digest preservation;
- promotion lineage failure on mismatched checkpoints.

## Scientific boundary

L19 reduces weight storage. It does not prove that int8 improves latency on every CPU or GPU.

The reference implementation dequantizes chunks into floating-point tensors before matrix multiplication. Native int8 kernels, hardware-specific packing and lower-bit formats are separate future optimizations and must earn their own quality/resource evidence.

L19 also does not change vocabulary/token semantics. A Vietnamese/English-focused tokenizer remains a later migration that should be evaluated only after real L18 rank selection and L19 quantization evidence are available.
