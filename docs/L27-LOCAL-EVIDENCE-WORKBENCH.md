# L27 Local Evidence Workbench

## Purpose

L24-L26 provide the individual pieces needed to create reviewed personal evidence:

- L24 imports a local conversation export into a never-approved queue;
- L26 records explicit human review decisions;
- L25 freezes those decisions into a verified L23 evidence pack;
- L21 checks whether the resulting pack is ready for the real candidate pipeline.

L27 combines those pieces into one local workspace without changing any approval boundary.

## State machine

```text
QUEUE_READY
    |
    | explicit human decisions only
    v
REVIEW_IN_PROGRESS
    |
    | all candidates decided
    v
REVIEW_COMPLETE
    |
    | explicit finalize
    v
INTAKE_READY
    |
    | readiness check only
    v
L21 REAL_CANDIDATE_INPUTS_READY / BLOCKED
```

No state transition creates approval automatically.

## Workspace

```text
local-evidence-workbench/
  review-queue/
    review-queue.jsonl
    review-queue-manifest.json
  review-decisions.jsonl
  review-progress-manifest.json
  intake/
    reviewed/
    approved-pack/
    local-evidence-intake-manifest.json
  workbench-manifest.json
```

The raw queue and approved dataset remain local-only by recommendation.
The workbench manifest contains only hashes, counts, relative paths and phase state.

## Initialize

```bash
python scripts/local_evidence_workbench.py init \
  --source /private/path/conversations.json \
  --workspace runtime-data/local-evidence-workbench
```

Initialization calls the real L24 queue builder.

A non-empty workspace is never overwritten.

## Review

```bash
python scripts/local_evidence_workbench.py review \
  --workspace runtime-data/local-evidence-workbench
```

This launches the real L26 reviewer against the workbench queue and decision file.

The workbench never synthesizes decisions.

## Status

```bash
python scripts/local_evidence_workbench.py status
```

Status verifies the queue and decisions before computing:

- total candidates;
- decided count;
- approved non-sensitive count;
- rejected count;
- sensitive count;
- remaining count;
- current workbench phase.

If a finalized L25 intake exists, status independently verifies its complete lineage before reporting `INTAKE_READY`.

## Finalize

By default:

```bash
python scripts/local_evidence_workbench.py finalize
```

requires:

- an explicit decisions file;
- zero undecided candidates;
- at least seven approved non-sensitive VI/EN examples.

The `--allow-undecided` switch is an explicit operator override. It still requires at least seven approved non-sensitive examples, and undecided candidates remain unapproved.

Finalize invokes the real L25 pipeline; it does not implement a second pack format.

## Readiness

```bash
python scripts/local_evidence_workbench.py readiness \
  --anchor research/personalization-general-anchor.jsonl \
  --latent runtime-data/living-core-shadow/latent.json \
  --model-lock model.lock.json \
  --model models/Qwen3-0.6B \
  --l14-anchor runtime-data/l14-minimal-anchor/minimal-qwen-anchor-cortex.pt
```

Readiness resolves the exact finalized L23/L25 approved pack and calls the same readiness assessor used by L21.

Authority remains:

```text
READINESS_ONLY_NO_TRAINING_AUTHORITY
```

L27 does not run `L21 --execute` automatically.

## Workbench manifest

The manifest binds:

- source export SHA-256 inherited from L24;
- L24 queue-manifest SHA-256;
- queue-file SHA-256;
- decisions-file SHA-256;
- review counts;
- current phase;
- L25 intake SHA-256 if finalized;
- L23 approved manifest SHA-256 if finalized;
- dataset SHA-256;
- protocol SHA-256;
- local relative paths.

It contains no raw prompt/target text and no candidate IDs.

## Courts

CI proves:

- a fresh workspace begins at `QUEUE_READY`;
- partial explicit review becomes `REVIEW_IN_PROGRESS`;
- complete explicit review becomes `REVIEW_COMPLETE`;
- finalize becomes `INTAKE_READY`;
- default finalize refuses undecided candidates;
- explicit allow-undecided still requires seven approved examples;
- fewer than seven approved examples fail closed;
- manifest tampering fails;
- decisions tampering after intake invalidates status;
- readiness uses the exact finalized approved pack;
- no private prompt/target sentinels appear in workbench/readiness receipts.

## Scientific boundary

L27 removes operational fragmentation, not the empirical bottleneck.

The first real model candidate still requires a real local conversation export, human review, a verified finalized intake, real Qwen/L14/latent inputs and a full L21 execution that passes all held-out quality and resource courts.
