# L25 Local Evidence Intake Pipeline

## Purpose

L24 creates a local review queue and requires explicit human decisions.
L23 converts an approved reviewed source into a frozen evidence pack.

L25 binds those two boundaries into one verifiable local lineage:

```text
L24 review queue manifest
          |
          +--> explicit decisions file
          |
          v
reviewed evidence source + manifest
          |
          v
L23 approved evidence pack
          |
          v
personalization.jsonl
personalization-protocol-v1.json
          |
          v
L21 readiness / real-candidate pipeline
```

L25 does not auto-approve anything and does not infer sensitivity.

## Finalize

```bash
python scripts/finalize_local_evidence_intake.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions /private/path/review-decisions.jsonl \
  --output-dir runtime-data/local-evidence-intake
```

The output directory contains:

```text
local-evidence-intake/
  reviewed/
    reviewed-evidence-source.jsonl
    reviewed-evidence-manifest.json
  approved-pack/
    personalization.jsonl
    personalization-protocol-v1.json
    approved-evidence-manifest.json
  local-evidence-intake-manifest.json
```

Raw reviewed text remains inside the local-only reviewed source and approved dataset.

## Intake manifest

The L25 manifest contains no raw prompts, targets, or source IDs.

It binds:

- L24 queue-manifest SHA-256;
- frozen queue-file SHA-256;
- explicit review-decisions SHA-256;
- reviewed-source manifest SHA-256;
- reviewed-source SHA-256;
- L23 approved-pack manifest SHA-256;
- personalization dataset SHA-256;
- frozen personalization protocol SHA-256;
- approved/rejected/sensitive/undecided counts;
- train/dev/test counts;
- VI/EN language counts.

Authority remains:

```text
EXPLICIT_REVIEWED_LOCAL_EVIDENCE_UNPROMOTED
```

This proves lineage, not model quality.

## Verification

```bash
python scripts/verify_local_evidence_intake.py \
  --manifest runtime-data/local-evidence-intake/local-evidence-intake-manifest.json \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions /private/path/review-decisions.jsonl
```

Verification independently rechecks:

- review-queue manifest integrity;
- queue-file integrity;
- decisions-file digest;
- reviewed-source manifest self-digest;
- reviewed-source digest;
- reviewed counts;
- L23 approved-pack manifest self-digest;
- dataset digest;
- frozen protocol/dataset binding;
- complete cross-stage lineage.

Any change to the queue, decisions, reviewed source, approved pack, dataset or protocol fails closed.

## L21 bridge

The approved manifest produced by L25 can be passed directly to L21:

```bash
python scripts/run_real_candidate_pipeline.py \
  --evidence-pack runtime-data/local-evidence-intake/approved-pack/approved-evidence-manifest.json
```

L21 still performs its own readiness checks. L25 does not bypass model, latent, anchor, L14 comparator, held-out quality or resource requirements.

## Courts

CI proves:

- seven approved non-sensitive rows + one sensitive + one rejected produce exactly seven training examples;
- the final manifest contains no private prompt/target/conversation identifiers;
- tampering with the decisions file after freeze fails;
- tampering with the reviewed source fails;
- fewer than seven eligible rows fail closed;
- existing output directories are never overwritten;
- the final approved pack drives L21 check-only successfully;
- the L21 receipt remains free of raw private sentinels.

## Scientific boundary

L25 closes the local evidence-intake lineage gap.

It does not determine whether reviewed examples are representative, correct, useful, balanced, or sufficient for a strong personal model. Human review and the downstream held-out courts remain authoritative.
