# L17 Factorized Language Boundary

## Purpose

L16 owns every inference tensor, but its largest tensors are still dense inherited language-boundary matrices.

L17 compresses those matrices into low-rank factors while leaving the Nolane recurrent cortex unchanged.

For an embedding matrix W with shape vocab x hidden:

```text
dense:
token id -> W[token]                         vocab x hidden

L17:
token id -> code[token] -> code @ basis      vocab x rank + rank x hidden
```

The output projection is factorized the same way. When input/output embeddings are tied, L17 reuses the same factors for both directions.

## Default rank

Default rank: **128**.

For a representative Qwen-style boundary with vocabulary 151,936 and hidden size 1,024:

- dense tied boundary: about 155.6M parameters;
- rank-128 factorized tied boundary: about 19.6M parameters;
- parameter ratio: about 12.6%.

The ~89.8K-parameter Nolane deep recurrent cortex is preserved separately and is not enlarged by vocabulary size.

## Initialization

L17 uses truncated SVD:

```text
W ~= U_r diag(S_r) V_r^T

codes = U_r diag(S_r)
basis = V_r^T
```

Small matrices use exact SVD. Large production matrices use randomized low-rank decomposition to avoid a full dense SVD.

The factorization receipt records:

- rank;
- input reconstruction error;
- output reconstruction error;
- dense parameter count;
- factorized parameter count;
- compression ratio;
- tied/untied contract.

## Boundary-only distillation

After factorization, the low-rank factors may be fine-tuned against L16 teacher logits.

Only factorized boundary parameters are trainable.

The Nolane cortex is frozen and its digest must remain unchanged.

```bash
python scripts/export_factorized_boundary.py --rank 128
python scripts/train_factorized_boundary.py
```

## Quality court

```bash
python scripts/evaluate_factorized_boundary.py
```

Default gates require:

- enough frozen held-out personal and VI/EN anchor evidence;
- factorized boundary <=35% of dense boundary parameters;
- held-out NLL regression versus L16 <=0.03;
- general-anchor NLL regression <=0.05;
- greedy-token agreement >=0.95;
- prompt full-scan == incremental carried-state scan;
- cortex digest identical to L16;
- no Qwen model runtime dependency;
- no Transformers model runtime dependency.

## Resource court

```bash
python scripts/benchmark_factorized_resources.py
```

Default gates require:

- checkpoint <=45% of L16 checkpoint size;
- boundary parameter ratio <=35%;
- forward latency <=1.20x L16;
- non-trivial generation throughput.

Promotion binds quality/resource evidence to the exact same L17 checkpoint and exact source L16 checkpoint.

```bash
python scripts/decide_factorized_promotion.py \
  --quality runtime-data/l17-quality.json \
  --resources runtime-data/l17-resources.json
```

## Neural court evidence

CI proves:

- full-rank factorization reconstructs dense input/output matrices within tight numerical tolerance;
- low-rank tied boundary reuses one factor pair for input and output;
- low-rank parameter count is analytically audited;
- factorized standalone runtime preserves prompt scan equivalence;
- boundary distillation changes boundary factors while the Nolane cortex receives zero gradients and keeps the same digest;
- factorized artifact roundtrip is independent of Qwen/Transformers runtime.

## Scientific boundary

L17 is more independent and much smaller, but it is still initialized from the L16/Qwen-derived language boundary and uses the inherited token-ID vocabulary.

It does not yet prove that a rank-128 real Qwen3-0.6B language boundary preserves production-quality Vietnamese/English generation.

A later wave may train a Vietnamese/English-focused native tokenizer and vocabulary boundary, but only after L17 establishes the compression-quality frontier.
