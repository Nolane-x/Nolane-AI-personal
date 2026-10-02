# L26 Interactive Local Reviewer

## Purpose

L24 creates a safe local review queue and L25 can turn explicit decisions into a hash-bound evidence pack.

L26 makes the human-review step practical without weakening the approval boundary.

It is a local terminal reviewer with no model or network dependency.

Opening a queue never approves anything.

## Run

```bash
python scripts/review_local_queue.py \
  --queue-manifest runtime-data/review-queue/review-queue-manifest.json \
  --decisions runtime-data/review-decisions.jsonl \
  --progress-manifest runtime-data/review-progress-manifest.json
```

For every pending user→assistant pair the reviewer shows the raw text locally and accepts exactly one of:

```text
a = approve, non-sensitive
r = reject, non-sensitive
s = reject as sensitive
k = skip / leave undecided
q = save current state and quit
```

There is no automatic approval command.

## Resume

Each completed decision is persisted immediately.

Restarting the command:

- verifies the frozen L24 queue;
- verifies every existing decision;
- rejects unknown candidate IDs;
- rejects duplicate decision rows;
- skips candidates that already have decisions;
- resumes only the remaining candidates.

A frozen decision cannot be silently changed through the session API. Reconsidering prior judgments requires an explicit new decisions file.

## Language boundary

An approved non-sensitive example must resolve to exactly:

```text
vi
or
en
```

If the imported conversation already has VI/EN metadata, L26 inherits it.

If it is missing, the terminal explicitly asks the reviewer before approval.

Rejected/sensitive/undecided examples never gain approval implicitly.

## Progress manifest

The progress manifest contains:

- queue-manifest SHA-256;
- queue-file SHA-256;
- decisions-file SHA-256 when decisions exist;
- total candidate count;
- decided count;
- approved non-sensitive count;
- rejected count;
- sensitive count;
- remaining count;
- self-digest.

Authority:

```text
HUMAN_REVIEW_SESSION_NO_AUTO_APPROVAL
```

It contains no raw prompt/target text and no candidate IDs.

Raw text is printed only to the local terminal during review.

## Atomic decisions

The decisions JSONL is rewritten atomically after every explicit decision.

This avoids a partially written decision file if the process stops during persistence.

Decision rows remain directly compatible with L24 `apply_review_decisions.py` and therefore with L25.

## Courts

CI proves:

- opening a queue creates zero decisions and zero approvals;
- progress manifest contains no raw prompt, target, conversation ID or candidate ID;
- each decision persists immediately;
- session resume exactly preserves prior decisions;
- an existing frozen decision cannot be silently replaced;
- unknown candidate IDs fail closed;
- duplicate existing decision rows fail closed;
- approval without resolvable VI/EN fails closed;
- progress-manifest tampering fails verification;
- reviewer decisions are directly consumed by L24;
- partial review leaves remaining candidates undecided.

## Scientific boundary

L26 improves the review workflow. It does not judge whether a conversation is high quality, representative, sensitive, or suitable for training.

Those judgments remain human decisions.

The first real candidate still requires enough manually reviewed L26 decisions, a verified L25 intake, and a full L21 execution that passes held-out quality and resource gates.
