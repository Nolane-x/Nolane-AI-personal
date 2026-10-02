# L21 Real Candidate Evidence Pipeline

## Purpose

L20 proves that the repository's factorization and int8 mechanics execute against the exact pinned Qwen3-0.6B weights.

L21 connects the already-built L15-L19 courts into one fail-closed evidence chain for a **real trained candidate**.

L21 does not invent training data and does not ship personal data in the repository.

A normal Git checkout is therefore expected to be **not ready** until the operator supplies the local evidence inputs.

## Readiness court

Run:

```bash
python scripts/assess_real_candidate_readiness.py
```

The readiness court checks only metadata and counts. It does not copy prompts or targets into its receipt.

Required inputs:

- frozen personalization dataset;
- frozen personalization protocol whose dataset SHA-256 verifies;
- train split with at least 1 example;
- dev split with at least 1 example;
- test split with at least 2 examples, matching downstream held-out courts;
- frozen general-language anchor with at least 4 examples;
- valid persistent 32D latent;
- exact pinned model weights;
- model revision marker matching `model.lock.json`;
- an L14 minimal-anchor candidate used as the L15 comparison baseline.

The receipt records:

- dataset SHA-256;
- protocol SHA-256;
- split counts;
- anchor count;
- latent digest and dimension, but never latent values;
- requested/resolved model revision;
- prerequisite presence flags.

It does **not** record personalization prompt/target text.

## Check-only mode

The full pipeline is safe by default:

```bash
python scripts/run_real_candidate_pipeline.py
```

Without `--execute`, it performs readiness only and writes:

```text
runtime-data/l21-real-candidate/pipeline-receipt.json
```

If inputs are sufficient the status is:

```text
REAL_CANDIDATE_READY_NOT_EXECUTED
```

If any prerequisite is absent or stale:

```text
REAL_CANDIDATE_PIPELINE_BLOCKED
```

## Full execution

Explicit execution:

```bash
python scripts/run_real_candidate_pipeline.py \
  --execute \
  --device cpu
```

For a CUDA machine:

```bash
python scripts/run_real_candidate_pipeline.py \
  --execute \
  --device cuda
```

A fresh workspace is mandatory. L21 refuses to mix new evidence into a non-empty workspace.

## Fail-closed stage chain

L21 runs exactly this authority chain:

```text
readiness
   |
   v
freeze L15 spec
   |
   v
train L15 native boundary
   |
   +--> L15 held-out quality
   +--> L15 resource court
   +--> L15 promotion
   |
   v
export L16 standalone
   |
   +--> L16 parity court
   +--> L16 resource court
   +--> L16 promotion
   |
   v
L18 adaptive rank search (train/dev only)
   |
   +--> held-out factorized quality
   +--> factorized resource court
   +--> selected-rank promotion
   |
   v
L19 int8 export
   |
   +--> held-out quantized quality
   +--> quantized resource court
   +--> quantized promotion
   |
   v
REAL_CANDIDATE_EVIDENCE_CHAIN_PASS
```

There are 17 executable stages after readiness.

If any script exits non-zero or fails to produce its expected artifact, L21 immediately records:

```text
REAL_CANDIDATE_PIPELINE_BLOCKED
blocked_stage = <exact stage>
```

No later stage runs.

## Evidence isolation

The pipeline receipt stores only:

- stage name;
- process exit code;
- expected output paths;
- SHA-256 of generated evidence/artifacts;
- top-level dataset/protocol/model lineage.

Subprocess output is not copied into the receipt.

The pipeline does not duplicate personal prompts, targets or latent vectors into its audit log.

## Promotion boundaries

L21 does not replace the underlying courts.

It invokes the existing frozen decisions:

- L15 native quality/resource/promotion;
- L16 standalone parity/resource/promotion;
- L18 rank-frontier quality/resource/promotion;
- L19 quantized quality/resource/promotion.

That means a later stage cannot hide an earlier failure.

For example:

- L15 cannot advance merely because it has zero Transformer decoder calls;
- L18 cannot advance merely because a smaller rank exists;
- L19 cannot advance merely because int8 uses fewer bytes.

Every wave must earn its own existing court result.

## Why L14 is an explicit prerequisite

The L15 quality court compares the native model with an L14 minimal-anchor candidate.

L21 deliberately does not manufacture a new L14 comparator inside the same execution and then treat it as external evidence.

The supplied L14 artifact is loaded by the existing L15 evaluator, which verifies its base-model and dataset fingerprint lineage.

A stale or unrelated L14 therefore fails closed.

## Current repository readiness

The repository intentionally does not commit:

- `runtime-data/personalization.jsonl`;
- the frozen user protocol;
- persistent latent state;
- trained L14/L15/L16/L18/L19 checkpoints;
- Qwen model weights.

Therefore CI proves the **readiness/pipeline mechanism**, not a fabricated production promotion.

Actual execution requires a local evidence workspace containing those user-approved inputs.

## Scientific boundary

L21 converts the project from a collection of individual experimental scripts into one reproducible promotion chain.

It does not create the missing empirical evidence.

The next meaningful milestone is a real L21 run on sufficient approved interaction data. The final status may legitimately be BLOCKED at any quality or resource gate; that is a useful result, not a pipeline failure.
