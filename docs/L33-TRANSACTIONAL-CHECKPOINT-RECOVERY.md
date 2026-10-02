# L33 Transactional Checkpoint Recovery

## Purpose

L31 can train one real continual-learning candidate.

L32 can prove a sequence of those candidates has valid ancestry and acceptable
long-horizon retention.

L33 protects the actual checkpoint transition itself.

The core requirement is simple:

> A crash, killed process, partial write or failed verification must never leave
> the runtime pointing at a half-installed model.

L33 therefore separates immutable model artifacts from one tiny serving pointer.

## Registry layout

A registry contains:

```text
registry/
  active.json
  .registry.lock
  artifacts/
    <checkpoint-sha256>/
  pointers/
    000000000000.json
    000000000001.json
    ...
  transactions/
    <transaction-id>/
      staging/
      events/
  rollbacks/
    <new-generation>.json
```

Model directories are immutable once installed.

The only mutable authority surface is `active.json`, and that file is replaced
atomically.

L33 still declares:

```text
TRANSACTIONAL_CHECKPOINT_SUBSTRATE_NO_AUTONOMOUS_PROD_AUTHORITY
```

so this mechanism is not itself permission for an autonomous model to update
production.

## Initialization

The initial factorized bundle is verified first:

- `factorized-nolane.pt` must exist;
- `factorized-nolane-manifest.json` must exist;
- checkpoint SHA-256 must match the manifest;
- artifact authority must remain unpromoted.

The bundle is copied into immutable `artifacts/<sha256>/`.

Generation 0 is then written into pointer history and atomically installed as
`active.json`.

## Candidate transaction

A new L31 candidate must contain:

- the factorized checkpoint;
- its factorized manifest;
- `l31-run-receipt.json`.

The L31 receipt is fully reverified through the L31/L30 lineage verifier.

In addition, L33 requires:

```text
candidate L31 parent checkpoint SHA
        ==
currently active checkpoint SHA
```

A valid candidate trained from some other parent cannot be silently inserted
into this registry.

## Staging

`begin` copies the candidate into transaction-local staging.

The transaction ID commits to:

- active parent pointer SHA;
- candidate checkpoint SHA;
- complete candidate bundle SHA;
- L31 run receipt SHA.

The first immutable event is:

```text
PREPARED
```

The staged copy is verified again after copying, which catches a source bundle
changing while the transaction is being prepared.

## Verification

`verify` requires the active pointer to still equal the transaction parent.

The staged candidate is rechecked for:

- checkpoint SHA drift;
- complete bundle drift;
- L31 receipt drift;
- L31/L30 evidence validity.

Only then is an immutable:

```text
VERIFIED
```

event appended.

## Atomic commit

Commit performs these steps:

1. recheck that the transaction parent is still active;
2. reverify the staged candidate;
3. install the candidate into the immutable artifact store;
4. create the next self-digested pointer snapshot;
5. atomically replace `active.json`;
6. append `POINTER_SWAPPED`;
7. append `COMMITTED`.

JSON authority files are written through a temporary file, flushed with
`fsync`, and then replaced with `os.replace`.

Directory metadata is also fsynced where the platform supports it.

## Crash before pointer swap

Suppose the process dies after PREPARED or VERIFIED but before step 5.

The active pointer is still the old parent.

Recovery sees:

```text
active pointer == transaction parent
```

and records:

```text
RECOVERED_ABORTED
```

The staging directory is removed.

An already copied but unreferenced immutable artifact may remain in the store,
but it has no serving authority.

## Crash after pointer swap

The harder case is a crash immediately after the atomic pointer replacement but
before `POINTER_SWAPPED` or `COMMITTED` is written.

Recovery does not guess.

It checks whether:

```text
active.transaction_id == interrupted transaction
and
active.checkpoint_sha256 == candidate checkpoint
```

If true, the atomic authority transition already happened.

Recovery therefore appends:

```text
RECOVERED_COMMITTED
```

instead of reverting a valid commit.

This distinguishes durable authority state from incomplete audit bookkeeping.

## Recovery conflict

If the active pointer is neither the transaction parent nor the transaction
candidate, automatic recovery refuses to choose.

It records:

```text
RECOVERY_CONFLICT
```

Manual investigation is required.

## Exclusive lock

Registry mutation uses an exclusive lock file created with
`O_CREAT | O_EXCL`.

A normal process always removes its lock.

A real process death can leave a stale lock. Recovery will not silently delete
it. The caller must explicitly request stale-lock breaking:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  recover --break-stale-lock
```

This prevents a second process from casually overriding a live writer.

## Rollback

Rollback never rewrites old pointer history.

If generation 1 is bad and generation 0 is selected, L33 creates generation 2:

```text
generation 0: model A
generation 1: model B
generation 2: model A   <- rollback transition
```

Generation 2 points to the already immutable model-A artifact and records:

- previous active pointer SHA;
- target generation;
- target checkpoint SHA;
- rollback ID;
- a self-digested rollback receipt.

This preserves the fact that model B was active and was later rolled back.

## Registry audit

The audit verifies:

- every pointer self-digest;
- exact generation sequence;
- pointer-to-pointer ancestry;
- every pointer's artifact exists;
- every artifact checkpoint SHA matches;
- active pointer equals the newest pointer snapshot;
- every rollback receipt self-digest;
- rollback receipt points to an actual rollback generation;
- rollback target generation agrees with the pointer.

Tampering with `active.json`, pointer history, artifact hashes or rollback
receipts makes the audit fail closed.

## CLI

Initialize:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  init --bundle /path/to/initial-factorized-bundle
```

Prepare and verify an L31 candidate:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  begin --candidate runtime-data/l31-cycle-next

python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  verify --transaction <transaction-id>
```

Commit:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  commit --transaction <transaction-id>
```

Recover after interruption:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  recover
```

Rollback:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  rollback --generation 0
```

Audit:

```bash
python scripts/manage_checkpoint_registry.py \
  --registry runtime-data/continual-checkpoint-registry \
  audit
```

## CI courts

L33 tests prove:

- a verified L31 candidate can commit atomically;
- wrong-parent L31 candidates are rejected;
- candidate staging tampering is detected;
- interruption before pointer swap recovers as aborted;
- interruption after pointer swap recovers as committed;
- rollback creates a new forward generation;
- stale locks require explicit recovery authority;
- active-pointer tampering is detected;
- rollback-receipt tampering is detected.

## Scientific and authority boundary

L33 makes checkpoint replacement crash-recoverable and auditable.

It still does not mean the model may autonomously decide to modify production.

A later authority layer must define exactly when real L32 multi-cycle evidence
is sufficient to request promotion, who or what authorizes that promotion, and
how serving processes reload a newly active checkpoint without split-brain
behavior.
