# L30 Long-Horizon Continual-Learning Court

## Purpose

L29 prevents a global held-out average from hiding a severe regression on one
independent source group.

L30 addresses a different failure mode: a model can look good immediately after
an update because it learns new evidence while quietly forgetting older learned
behavior.

The L30 court evaluates a **sequential checkpoint transition**:

```text
checkpoint before update
        |
        | train/update only on new approved evidence
        v
checkpoint after update
```

Both checkpoints are then measured on two frozen, disjoint evidence sets:

```text
retention set  = older evidence that must remain stable
adaptation set = newly introduced evidence that should improve
```

The court is deliberately model-agnostic. It consumes per-example,
lower-is-better metrics such as NLL measured on the exact same examples before
and after the update.

## What L30 proves

A PASS means the measured checkpoint transition simultaneously satisfies:

- a real checkpoint change occurred;
- retention and adaptation source groups are disjoint;
- enough independent source groups exist in both courts;
- mean retention regression stays within policy;
- the worst old source group stays within the forgetting limit;
- the new evidence achieves the required average gain;
- no new source group regresses beyond the adaptation limit.

This is a stability/plasticity court. It is not a consciousness claim and it
does not by itself grant production authority.

## Why both mean and worst-group gates exist

A mean-only forgetting metric is unsafe.

Example:

```text
old group A regression = -0.10
old group B regression =  0.00
old group C regression = +0.18

overall examples ~= stable
worst group      = badly forgotten
```

L30 therefore gates both the mean and the worst independent retention group.

The same rule is applied to adaptation. A strong gain on two new groups cannot
hide a severe regression on another new group.

## Evidence contract

The CLI accepts a local JSON document:

```json
{
  "schema": "NOLANE-L30-CONTINUAL-EVIDENCE-V1",
  "pre_update_checkpoint_sha256": "<sha256>",
  "post_update_checkpoint_sha256": "<sha256>",
  "retention": {
    "group_sha256": ["<sha256>", "<sha256>"],
    "before_values": [1.10, 0.92],
    "after_values": [1.11, 0.93]
  },
  "adaptation": {
    "group_sha256": ["<sha256>", "<sha256>"],
    "before_values": [1.40, 1.35],
    "after_values": [1.18, 1.10]
  }
}
```

The input remains local. The receipt never copies source-group hash values.

Run:

```bash
python scripts/assess_continual_learning.py \
  --evidence runtime-data/l30/continual-evidence.json \
  --output runtime-data/l30/continual-learning-receipt.json
```

## Default policy

The default policy is intentionally conservative:

- retention groups: at least 2;
- adaptation groups: at least 2;
- worst retention regression: at most +0.03;
- mean retention regression: at most +0.01;
- mean adaptation gain: at least 0.00;
- worst adaptation regression: at most +0.03.

Individual training stages may tighten these limits, but should not silently
weaken them while claiming the same authority.

## Privacy

The output receipt contains:

- checkpoint digests;
- counts;
- local group aliases;
- split-local indices;
- before/after means;
- regressions and gains;
- policy;
- court digest.

It does not contain:

- prompts;
- targets;
- raw conversation IDs;
- source-group SHA values.

## Fail-closed conditions

L30 BLOCKS on:

- same before/after checkpoint digest;
- insufficient independent old groups;
- insufficient independent new groups;
- any retention/adaptation source-group overlap;
- non-finite metrics;
- metric/group length mismatch;
- excessive mean forgetting;
- excessive worst-group forgetting;
- insufficient average adaptation gain;
- excessive regression on any new group.

The receipt is self-digested. Tampering invalidates the court.

## Relationship to current architecture

L30 does not modify Qwen, L15, L16, L17, L18 or L19 directly.

It creates the court required before future **continual update promotion** can be
considered scientifically meaningful. A future updater must produce a real
pre-update checkpoint and a real post-update checkpoint, then pass L30 on
frozen old/new evidence before receiving any production authority.

This avoids a common mistake: declaring "continual learning" merely because a
model can be fine-tuned repeatedly.

## Courts in CI

The unit court proves:

- stable old groups + useful new learning can PASS;
- an acceptable global retention average cannot hide one badly forgotten old
  group;
- average new-evidence gain cannot hide one badly regressed new group;
- old/new source overlap blocks;
- an unchanged checkpoint blocks;
- non-finite or length-drift evidence is rejected;
- source-group hashes do not leak into receipts;
- receipt tampering is detected;
- promotion-status extraction fails closed when the L30 receipt is absent,
  blocked or tampered.

## Scientific boundary

L30 measures catastrophic-forgetting risk for one explicit sequential update.

It does not yet prove indefinite lifelong learning. That requires repeated
multi-update runs over long time horizons, rollback/recovery tests and a
production updater whose authority is bound to these receipts.
