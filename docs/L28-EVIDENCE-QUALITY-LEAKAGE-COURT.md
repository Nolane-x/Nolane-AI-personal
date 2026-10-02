# L28 Evidence Quality & Leakage Court

## Purpose

A held-out score is only meaningful if the held-out examples are actually independent of training evidence.

L28 adds a deterministic structural court before any real-candidate training may be considered ready.

It does not judge whether a conversation is emotionally good, personally representative, or semantically valuable. It only checks measurable evidence integrity.

## Source-group lineage

L24 already records a local `source_id` for every extracted user→assistant pair.

L23 now derives two privacy-preserving hashes:

```text
source_id_sha256
source_group_sha256
```

For L24 data, a source group is the conversation prefix before `:pair:<index>`.

The raw conversation ID is never copied into the approved-pack manifest, protocol court receipt, L25 intake manifest, L27 workbench manifest, or L21 receipt.

## Group-aware split

When every approved example has source-group lineage, the personalization protocol uses:

```text
source_group_chronological_v1
```

All examples from one source group are assigned to exactly one of:

```text
train
dev
test
```

The splitter searches deterministic group boundaries that best approximate the requested train/dev/test proportions while keeping every split non-empty.

A dataset with fewer than three distinct source groups cannot produce a leakage-safe three-way split.

## Structural court

Run it directly:

```bash
python scripts/assess_evidence_quality.py \
  --dataset runtime-data/personalization.jsonl \
  --protocol runtime-data/personalization-protocol-v1.json
```

The court checks:

- at least seven total examples for the real-candidate path;
- non-empty train/dev and at least two held-out test examples;
- complete per-example source-group lineage;
- at least three distinct source groups;
- grouped split strategy;
- zero source-group overlap across train/dev/test;
- zero exact normalized prompt leakage across splits;
- zero exact prompt+target leakage across splits;
- near-duplicate prompt+target pairs across different splits;
- deterministic dataset/protocol identity.

Near-duplicate detection requires both:

```text
token Jaccard >= 0.80
sequence similarity >= 0.90
```

for pairs with enough lexical content. This avoids treating short unrelated strings as duplicates while still catching lightly edited copies.

## Quality receipt

A PASS receipt contains only structural metrics:

- dataset SHA-256;
- protocol SHA-256;
- policy;
- split counts;
- source-group coverage/counts;
- leakage counts;
- near-duplicate index pairs and similarity scores;
- language counts per split;
- diversity ratios;
- court SHA-256.

It contains no raw prompts, targets, raw source IDs, or source-group hash values.

Authority:

```text
STRUCTURAL_QUALITY_ONLY_NO_MODEL_AUTHORITY
```

## L23 integration

L23 now builds the dataset, grouped protocol and L28 receipt in memory before creating the output pack directory.

If L28 is BLOCKED, no complete approved pack is written.

A successful pack contains:

```text
personalization.jsonl
personalization-protocol-v1.json
evidence-quality-receipt.json
approved-evidence-manifest.json
```

The L23 manifest binds:

- quality status;
- quality court SHA-256;
- quality receipt file SHA-256.

Verification recomputes the court from the current dataset/protocol and rejects any mismatch.

## L25 / L27 lineage

L25 binds the L28 quality court into the local intake manifest.

L27 surfaces the same quality status and court SHA in the workbench state and readiness receipt.

Any post-finalization decisions/dataset/protocol/quality-receipt drift invalidates the corresponding lineage.

## L21 readiness integration

`assess_real_candidate_readiness` recomputes L28 directly from the dataset and protocol.

This means the quality gate cannot be bypassed by passing `--dataset` and `--protocol` directly instead of using `--evidence-pack`.

A legacy or generic protocol without per-example source-group lineage may still be structurally valid for non-production experiments, but it is BLOCKED for the L21 real-candidate path with reasons such as:

```text
evidence_quality:incomplete_source_group_lineage
evidence_quality:insufficient_distinct_source_groups
evidence_quality:non_grouped_split_strategy
```

## Courts

CI proves:

- grouped splitting keeps one conversation entirely inside one split;
- fewer than three source groups fail closed;
- missing source-group lineage blocks quality/readiness;
- near-duplicate evidence crossing train/dev is detected;
- L23 emits a verified quality receipt;
- raw private prompt/target/conversation sentinels never enter the quality receipt or manifest;
- quality-receipt tampering is detected;
- L21 direct dataset/protocol check-only requires L28 PASS;
- L25/L27/L21 receipts preserve the quality court SHA lineage.

## Scientific boundary

L28 protects the validity of held-out evidence.

It does not prove that seven examples are enough to train a strong personal model, that the data represent the user's full personality, or that a candidate is better than Qwen.

Those claims still require the downstream L21 empirical quality and resource courts.
