# L23 Approved Evidence Pack Builder

## Purpose

L21 can execute the real candidate evidence chain, but it intentionally refuses to invent or commit personal training evidence.

L23 adds the missing local-only intake boundary.

A user may prepare a JSONL source in which every row must explicitly declare whether it is approved for model development. L23 filters that source, freezes a train/dev/test protocol, and emits a hash-bound evidence pack that L21 can consume directly.

L23 does not scrape chats automatically and does not infer consent.

## Source format

Example local source:

```json
{"prompt":"Hôm nay tôi hơi mệt.","target":"Ừ, nay ông có vẻ xuống pin thật. Nghỉ tí đi.","language":"vi","approved":true,"sensitive":false,"source_id":"conversation-001"}
{"prompt":"Could you remind me what I said about the exam?","target":"You said you were worried about the math exam.","language":"en","approved":true,"sensitive":false,"source_id":"conversation-002"}
{"prompt":"Do not use this one.","target":"Private.","language":"en","approved":false}
```

Only rows with:

```text
approved == true
AND
sensitive != true
```

are eligible.

No implicit approval exists.

## Default policy

- languages: Vietnamese and English;
- minimum approved non-sensitive examples: 7;
- exact prompt/target duplicates: rejected;
- prompt length <= 8,000 characters;
- target length <= 8,000 characters;
- weight range: 0.25 to 4.0;
- output directory must be empty or absent.

Seven is the minimum because the default frozen split must leave at least:

- train >=1;
- dev >=1;
- held-out test >=2.

For exactly seven examples the default split is 4 train / 1 dev / 2 test.

## Build

```bash
python scripts/build_approved_evidence_pack.py \
  --source /private/path/approved-conversations.jsonl \
  --output-dir runtime-data/approved-evidence
```

The resulting directory contains:

```text
runtime-data/approved-evidence/
  personalization.jsonl
  personalization-protocol-v1.json
  approved-evidence-manifest.json
```

`runtime-data/` is gitignored by the repository.

## Privacy boundary

The dataset and frozen protocol necessarily contain the approved text and therefore should remain local.

The manifest does not contain:

- raw prompts;
- raw targets;
- raw source IDs.

Instead it records:

- source-file SHA-256;
- SHA-256 hashes of optional source IDs;
- dataset SHA-256;
- protocol SHA-256;
- counts and language distribution;
- policy;
- split counts.

Manifest authority is:

```text
USER_APPROVED_LOCAL_EVIDENCE_UNPROMOTED
```

Approval makes data eligible for the empirical pipeline. It does not promote a model.

Rows with `sensitive:true` are always excluded even if `approved:true`.

## Verification

```bash
python scripts/verify_approved_evidence_pack.py \
  --manifest runtime-data/approved-evidence/approved-evidence-manifest.json
```

Verification checks:

- manifest digest;
- dataset digest;
- frozen personalization protocol digest;
- protocol/dataset binding;
- example count.

Any dataset edit after freezing fails closed.

## L21 integration

The verified pack can be consumed directly:

```bash
python scripts/assess_real_candidate_readiness.py \
  --evidence-pack runtime-data/approved-evidence/approved-evidence-manifest.json

python scripts/run_real_candidate_pipeline.py \
  --evidence-pack runtime-data/approved-evidence/approved-evidence-manifest.json
```

The L21 receipt binds:

- approved evidence manifest SHA-256;
- evidence-pack authority;
- dataset SHA-256;
- protocol SHA-256;
- canonical 17-stage L22 contract SHA-256.

It does not copy approved prompt/target text into the receipt.

Full training still requires the existing non-data prerequisites such as the pinned model, persistent latent, general-language anchor and valid L14 comparator.

## Courts

CI verifies:

- unapproved rows are excluded;
- approved but sensitive rows are excluded;
- exact duplicates fail closed;
- unsupported languages fail closed;
- fewer than seven eligible rows fail closed;
- non-empty output directories are never overwritten;
- dataset tampering fails verification;
- manifest contains no raw private prompt/target or source IDs;
- generated pack is compatible with L21 readiness;
- `--evidence-pack` drives L21 check-only successfully;
- the L21 receipt binds the exact manifest SHA while remaining free of private sentinels.

## Scientific boundary

L23 solves evidence intake and consent lineage. It does not create evidence that the user has not explicitly supplied and approved.

The next real milestone is an actual local L21 `--execute` run using a sufficient L23 pack, a valid persistent latent, the pinned Qwen checkpoint and the matching L14 comparator. A quality/resource gate may still block, and that is an empirical result rather than an orchestration failure.
