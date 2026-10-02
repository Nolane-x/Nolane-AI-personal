# L24 Local Review Queue

## Purpose

L23 accepts only explicitly approved local examples.

L24 adds a safer path from an existing conversation export to that approval boundary.

Importing a conversation does **not** approve it.

Every imported user→assistant pair enters a local review queue with:

```json
{
  "approved": false,
  "reviewed": false,
  "sensitive": null
}
```

A separate decision file is required before anything can become approved evidence.

## Generic conversation export

L24 intentionally uses a small generic schema instead of assuming a vendor-specific export format.

JSON array example:

```json
[
  {
    "conversation_id": "local-conversation-001",
    "language": "vi",
    "messages": [
      {"role": "user", "content": "Hôm nay tôi hơi mệt."},
      {"role": "assistant", "content": "Ừ, nay nghỉ sớm một chút đi."}
    ]
  }
]
```

JSONL containing one conversation object per line is also supported.

Only `user` and `assistant` roles are considered. System/tool messages are skipped.

Pairs are formed from a pending user turn followed by an assistant turn.

## Build the local queue

```bash
python scripts/build_review_queue.py \
  --source /private/path/conversations.json \
  --output-dir runtime-data/review-queue
```

If the export does not provide a language:

```bash
python scripts/build_review_queue.py \
  --source /private/path/conversations.json \
  --default-language vi
```

The queue directory contains:

```text
review-queue.jsonl
review-queue-manifest.json
```

The queue contains raw local text and should remain private.

The manifest contains hashes/counts only and never stores raw prompt/target or raw conversation IDs.

## No automatic consent

Every imported row is forced to:

```text
approved = false
reviewed = false
sensitive = null
```

Verification fails if an imported queue has already been changed to `approved:true`.

That means editing the queue itself is not a valid approval path.

Approval must be expressed through a separate decisions file.

## Decision file

Example:

```json
{"candidate_id":"<id>","approved":true,"sensitive":false,"language":"vi"}
{"candidate_id":"<id>","approved":false,"sensitive":true,"language":"en"}
```

Both `approved` and `sensitive` must be explicit booleans for every supplied decision.

Apply decisions:

```bash
python scripts/apply_review_decisions.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions /private/path/review-decisions.jsonl \
  --output-dir runtime-data/reviewed-evidence
```

Unknown candidate IDs fail closed.

Duplicate decisions for the same candidate fail closed.

Candidates without a decision remain:

```text
approved = false
reviewed = false
```

An approved non-sensitive candidate must have an explicit or inherited `vi`/`en` language.

## Feed L23

The reviewed source can be passed directly to L23:

```bash
python scripts/build_approved_evidence_pack.py \
  --source runtime-data/reviewed-evidence/reviewed-evidence-source.jsonl \
  --output-dir runtime-data/approved-evidence
```

L23 then applies its own independent rules again:

- only `approved:true`;
- exclude `sensitive:true`;
- reject duplicates;
- enforce VI/EN;
- freeze train/dev/test;
- bind dataset/protocol/manifest hashes.

Thus L24 review does not bypass L23.

## Integrity

The queue manifest binds:

- source export SHA-256;
- queue SHA-256;
- candidate count;
- extraction policy.

The reviewed-evidence manifest binds:

- queue manifest SHA-256;
- decisions-file SHA-256;
- reviewed source SHA-256;
- approved/rejected/sensitive/undecided counts.

Changing the queue after freezing makes queue verification fail.

## Privacy boundary

Raw text exists only in local queue/reviewed-source files.

Manifests contain no raw prompt, target or conversation ID.

All default output paths live under gitignored `runtime-data/`.

## Courts

CI proves:

- imports always start `approved:false`;
- imports always start `reviewed:false`;
- system/tool messages do not become examples;
- only user→assistant pairs are extracted;
- queue manifest contains no private sentinels;
- queue tampering cannot turn import into approval;
- partial decisions leave untouched candidates unapproved;
- unknown candidate IDs fail closed;
- duplicate decisions fail closed;
- approved candidates without VI/EN language fail closed;
- exact duplicate conversation pairs fail closed;
- reviewed source passes through L23 and only approved non-sensitive rows survive.

## Scientific boundary

L24 improves the human approval workflow. It does not decide whether a conversation is useful, sensitive, correct or representative.

Those judgments remain human decisions.

The first actual trained candidate still requires a sufficient reviewed L23 pack and the remaining L21 prerequisites.
