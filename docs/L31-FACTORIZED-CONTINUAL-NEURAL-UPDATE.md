# L31 Factorized Continual Neural Update

## Purpose

L30 defines the stability/plasticity court for one sequential checkpoint
transition.

L31 connects that court to a real trainable neural path.

The target is the standalone L17/L18 factorized language boundary. L31 starts
from one frozen parent factorized checkpoint, loads it twice, and gives the two
copies different roles:

```text
parent factorized checkpoint
        |
        +-------------------+
        |                   |
        v                   v
frozen reference       trainable candidate
        |                   |
        |              low-rank boundary only
        |                   |
        +-------- L30 -------+
```

The deep recurrent Nolane cortex is frozen on both sides. The frozen reference
is never optimized.

## Four evidence partitions

L31 deliberately separates update data from court data.

```text
old protocol
  train + dev -> retention rehearsal
  test        -> retention held-out court

new protocol
  train       -> adaptation update
  test        -> adaptation held-out court
```

The old held-out test rows are not used as rehearsal examples.

The new held-out test rows are not used to update the candidate.

This prevents the update from earning a continual-learning PASS by being scored
on the same examples it just optimized.

## Update objective

Every adaptation step is paired round-robin with one old rehearsal example.

The candidate loss contains:

- task loss on the new adaptation row;
- optional task loss on the old rehearsal row;
- KL distillation on the old rehearsal row against the frozen parent.

Only low-rank factorized boundary parameters are trainable.

The final normalization copied from the parent remains frozen.

The recurrent cortex must receive zero gradients.

## Parent invariants

Before the first optimizer step:

- candidate boundary digest must exactly equal reference boundary digest;
- candidate cortex digest must exactly equal reference cortex digest;
- candidate and reference must be distinct model objects.

After training:

- candidate boundary is expected to change;
- candidate cortex must remain unchanged;
- reference boundary must remain unchanged;
- reference cortex must remain unchanged.

Violation of the neural ownership boundary blocks the run.

## Evidence quality before training

Both the old and new datasets must independently pass the L28 structural
quality/leakage court.

That means the sequential updater refuses to train from a protocol with:

- incomplete source-group lineage;
- train/dev/test group overlap;
- exact cross-split prompt or pair leakage;
- high-similarity cross-split near duplicates;
- insufficient evidence for the current L28 policy.

## L30 after training

The frozen parent and trained candidate are evaluated on:

- old held-out retention groups;
- new held-out adaptation groups.

L30 then applies its stability/plasticity policy:

- mean forgetting;
- worst-group forgetting;
- mean adaptation gain;
- worst-group adaptation regression.

A trained candidate may exist while the L30 status is `BLOCKED`. Training
success is never treated as promotion success.

## Local real-data runner

```bash
python scripts/train_continual_factorized_update.py \
  --factorized runtime-data/l17-factorized-trained/factorized-nolane.pt \
  --retention-dataset /private/old/personalization.jsonl \
  --retention-protocol /private/old/personalization-protocol-v1.json \
  --adaptation-dataset /private/new/personalization.jsonl \
  --adaptation-protocol /private/new/personalization-protocol-v1.json \
  --output-dir runtime-data/l31-continual-factorized-update
```

The output workspace is fail-closed: an existing non-empty directory is never
overwritten.

## Lineage

The saved candidate binds:

- parent factorized checkpoint SHA-256;
- old dataset SHA-256;
- old protocol SHA-256;
- old L28 court SHA-256;
- new dataset SHA-256;
- new protocol SHA-256;
- new L28 court SHA-256;
- one combined lineage SHA-256;
- embedded L30 continual-learning receipt.

Raw prompts, targets and source-group hashes are not copied into the court
receipt.

## Production boundary

L31 emits:

```text
CONTINUAL_UPDATE_EVIDENCE_ONLY_UNPROMOTED
```

It does not modify the production model pointer.

A candidate is not accepted merely because:

- optimizer steps completed;
- loss decreased;
- the boundary changed;
- new data improved.

The run returns success only when the ownership invariants remain intact and
the L30 held-out continual-learning court passes.

## CI courts

Tiny real-PyTorch courts verify:

- only the candidate factorized boundary changes;
- the candidate recurrent cortex remains unchanged;
- the frozen reference remains unchanged;
- boundary gradients are observed;
- cortex gradients are absent;
- L30 evidence is embedded;
- source-group hashes do not leak into receipts;
- a candidate can train yet still be BLOCKED by a deliberately strict L30
  policy;
- mismatched parent state is rejected;
- candidate/reference aliasing is rejected;
- lineage-length drift is rejected.

## Scientific boundary

L31 is the first real sequential neural-update path in this repository that is
bound to old/new held-out continual-learning evidence.

It still does not prove indefinite lifelong learning.

That requires a later multi-cycle authority where checkpoint N becomes the
frozen parent of checkpoint N+1 repeatedly, with rollback, interruption
recovery, bounded cumulative drift and long-horizon retention measured across
multiple historical windows.
