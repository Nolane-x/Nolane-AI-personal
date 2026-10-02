# L20 Real Qwen3-0.6B Weight Court

## Purpose

L5-L19 use increasingly strong structural and neural courts, but most automated architecture tests intentionally use tiny Qwen3 instances so CI stays cheap.

L20 adds a separate heavy court that downloads and executes the exact pinned **Qwen/Qwen3-0.6B** checkpoint.

This is the first court in this repository that verifies the L17/L19 language-boundary mechanics against learned weights taken directly from the full pinned checkpoint.

It is still a **mechanical evidence court**, not a quality promotion.

## Exact upstream identity

The court uses:

- repository: `Qwen/Qwen3-0.6B`
- revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- model type: `qwen3`

The pinned downloader verifies that Hugging Face resolves the requested revision to that exact commit.

L20 also corrected an old metadata error in `model.lock.json`.

Measured directly from the pinned model:

**596,049,920 parameters**

The previous lock value `751,632,384` was rejected by the court.

## What the real-weight court does

```text
download exact pinned checkpoint
          |
          v
load full Qwen3-0.6B
          |
          +--> count actual parameters
          +--> inspect vocab/hidden/depth/tied boundary
          +--> execute a real full-model forward
          |
          v
sample 512 deterministic vocabulary rows
from the learned input/output boundary
          |
          v
rank-128 factorization
          |
          v
row-wise int8 quantization
          |
          v
mechanical fidelity + full-shape storage court
```

The full model is released from memory before the SVD/int8 phase.

The sampled SVD is deliberate: the goal of L20 is to test code and learned-weight behavior on the real checkpoint without pretending that a CPU CI runner has trained or validated a full rank-128 production boundary.

Full-vocabulary parameter/storage numbers are computed from the **actual measured Qwen shapes**.

## Successful authority run

Successful run:

- workflow run: **36977448315**
- head: **`bb2ff92e5af170f7e618bf69ff910d615f622b3f`**
- artifact: **11214325349**
- artifact digest: **`sha256:9754324c4be62c5445445fc2ba19a1ab20234bd1d978195c8508e67a5642896c`**
- status: **REAL_QWEN06_WEIGHT_COURT_PASS**

The workflow is:

```text
.github/workflows/real-qwen06-weight-court.yml
```

It can also be invoked manually through GitHub Actions.

## Observed real Qwen3-0.6B structure

The successful court measured:

- parameters: **596,049,920**
- decoder layers: **28**
- hidden size: **1,024**
- vocabulary: **151,936**
- input/output embeddings: **tied**
- source boundary storage: **311,166,976 bytes**
- real full-model forward: finite logits
- test forward sequence length: **6 tokens**

The top token ID from the deterministic smoke forward was 358. This is only a reproducibility signal, not a quality metric.

## Rank-128 real-weight sample

The court samples 512 deterministic vocabulary rows from the learned boundary and applies the same L17 factorization code.

Observed rank-128 sample reconstruction error:

- input: **0.6398504**
- output: **0.6398504**

That number is intentionally **not** a promotion gate. Rank truncation is lossy before distillation, and a 512-row mechanical sample cannot establish production language quality.

It does prove that the production factorization code executes on actual learned Qwen3-0.6B boundary rows.

## Int8 fidelity on real learned factors

After factorization, the L19 row-wise int8 path produced:

- input factor quantization error: **0.0112846**
- output factor quantization error: **0.0112846**
- factorized-vs-int8 probe-logit relative error: **0.0110695**

The mechanical gate allows at most 0.03 factor error and 0.06 probe-logit error.

These gates passed.

## Full-boundary analytical footprint from measured shapes

Using the measured real Qwen boundary dimensions and rank 128:

- dense boundary parameters: **155,583,488**
- factorized boundary parameters: **19,579,904**
- parameter ratio: **0.1258482** (~12.58%)
- factorized FP32 storage: **78,319,616 bytes**
- factorized BF16/FP16 storage: **39,159,808 bytes**
- L19 int8+scale storage: **20,191,232 bytes**
- int8 / FP32-factorized ratio: **0.2578056**
- int8 / BF16-factorized ratio: **0.5156111**

This passes the L19 byte-level gates:

- <30% of FP32 factorized storage;
- <60% of BF16/FP16 factorized storage.

## Run locally

After downloading the pinned checkpoint:

```bash
python scripts/download_model.py

python scripts/run_real_qwen06_weight_court.py \
  --model models/Qwen3-0.6B \
  --sample-rows 512 \
  --rank 128 \
  --output runtime-data/l20-real-qwen06-court.json
```

## What L20 proves

L20 proves that:

- the repository can reproduce the exact pinned upstream checkpoint;
- the real model shape matches the architecture assumptions used by L16-L19;
- a full Qwen3-0.6B forward executes under the court;
- L17 factorization code executes on real learned boundary weights;
- L19 row-wise int8 code preserves those factorized weights/logits within the frozen mechanical thresholds;
- full-boundary storage calculations are now grounded in measured real dimensions rather than only representative constants;
- metadata drift such as the old parameter-count error fails closed.

## What L20 does not prove

L20 does **not** prove that:

- an undistilled rank-128 full boundary preserves language quality;
- L18 will select rank 128 on real train/dev evidence;
- L19 quantized generation passes held-out Vietnamese/English quality;
- the standalone Nolane recurrent cortex already matches full Qwen3-0.6B reasoning;
- int8 storage automatically means faster inference.

Those claims remain blocked behind the existing train/dev/test and target-device resource courts.

The next scientifically useful step is therefore to obtain a real L16/L18 trained candidate and run the full held-out frontier, not to infer production quality from this mechanical PASS.
