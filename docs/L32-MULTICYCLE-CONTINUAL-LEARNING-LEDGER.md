# L32 Multi-Cycle Continual Learning Ledger

## Purpose

L31 proves one sequential neural update can be trained and judged on held-out
old/new evidence.

L32 asks the next question:

> Can multiple L31 updates form one auditable learning history without silently
> breaking checkpoint lineage or accumulating long-horizon forgetting?

L32 combines two independent courts:

1. an exact multi-cycle artifact/state chain;
2. a fixed held-out retention panel comparing the final checkpoint directly to
   the checkpoint that existed before the first update.

Both must pass.

## Why per-cycle PASS is not enough

Suppose every individual update stays within a small forgetting budget.

That still does not prove the final checkpoint remembers behavior from the
beginning of the learning history. Small losses can accumulate.

L32 therefore keeps the per-cycle L30 courts but adds one direct endpoint test:

```text
checkpoint 0 -------------------------------> checkpoint N
     |                                             |
     +------ same frozen held-out panel -----------+
```

The final model is compared directly with the initial model on the exact same
fixed panel.

## Cycle receipt

L31 now persists its complete run receipt by default:

```text
<output-dir>/l31-run-receipt.json
```

The receipt binds:

- parent factorized checkpoint SHA-256;
- candidate boundary state before/after;
- immutable recurrent cortex state;
- old/new L28 quality courts;
- old/new dataset + protocol lineage;
- L30 continual-learning court;
- saved candidate artifact SHA-256.

L32 does not accept a summarized hand-written record in place of this receipt.

## Exact chain requirements

For every adjacent pair of cycles:

```text
cycle i artifact checkpoint SHA
        ==
cycle i+1 parent checkpoint SHA
```

and:

```text
cycle i candidate boundary digest after
        ==
cycle i+1 candidate boundary digest before
```

The first equality proves file-level artifact ancestry.

The second equality proves neural-state continuity.

Every embedded L30 court must be self-digest valid and PASS.

Every saved artifact must contain the exact L31 training receipt that the run
receipt claims, and its boundary/cortex digests must match the trained state.

## Fresh adaptation windows

By default every cycle must use a different adaptation protocol SHA-256.

Replaying one already-used adaptation protocol does not count as a new learning
cycle.

This rule can be disabled only explicitly in the CLI for diagnostic runs; such
a diagnostic should not be presented as fresh continual-learning evidence.

## Fixed long-horizon retention panel

The fixed panel is a leakage-safe L28-approved personalization protocol.

Only its test split is used.

The evaluator loads the initial and final factorized checkpoints and requires:

- same L16 ancestry;
- identical recurrent-cortex digest;
- identical factorized boundary configuration;
- valid frozen personalization protocol;
- L28 PASS on the panel.

It then measures per-example NLL on the same test examples for both endpoints.

The long-horizon court gates:

- overall final-minus-initial NLL regression;
- worst independent source-group regression;
- at least two independent held-out source groups.

The receipt uses local group aliases and never copies source-group hashes.

## Real endpoint evaluator

```bash
python scripts/evaluate_long_horizon_retention.py \
  --initial /path/to/checkpoint-0/factorized-nolane.pt \
  --final /path/to/checkpoint-N/factorized-nolane.pt \
  --dataset /private/fixed-panel/personalization.jsonl \
  --protocol /private/fixed-panel/personalization-protocol-v1.json \
  --output runtime-data/l32/long-horizon-retention.json
```

Default limits:

- overall regression <= +0.01 NLL;
- worst source-group regression <= +0.03 NLL.

The limits are explicit policy, not hidden heuristics.

## Multi-cycle verifier

After at least two L31 cycles:

```bash
python scripts/assess_multicycle_continual.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l31-run-receipt.json \
  --long-horizon runtime-data/l32/long-horizon-retention.json \
  --output runtime-data/l32/multicycle-chain.json
```

The fixed-panel initial checkpoint must equal cycle 1's parent artifact.

The fixed-panel final checkpoint must equal the final cycle's saved artifact.

## What the chain summary records

Without copying private examples or source identifiers, L32 records:

- number of cycles;
- first parent artifact SHA-256;
- final artifact SHA-256;
- each cycle's before/after neural boundary digests;
- each L30 court SHA-256;
- each L31 lineage SHA-256;
- per-cycle mean and worst retention regression;
- per-cycle adaptation gain;
- sum of positive mean retention regressions;
- worst observed cycle-level retention regression;
- number of unique adaptation protocols;
- fixed-panel long-horizon court SHA-256.

The moving-window statistics are diagnostic.

The fixed endpoint panel is the stronger cumulative-forgetting evidence.

## Fail-closed conditions

L32 blocks when:

- fewer than two cycles are supplied;
- an L31 run schema or authority is wrong;
- an embedded L31 lineage digest is invalid;
- an embedded L30 receipt is missing, tampered or BLOCKED;
- the candidate/reference neural ownership invariants were violated;
- the artifact does not contain the exact claimed training receipt;
- saved boundary/cortex digests do not match the training receipt;
- artifact SHA ancestry breaks;
- neural boundary-state continuity breaks;
- an adaptation protocol is reused when uniqueness is required;
- the fixed-panel receipt is invalid or BLOCKED;
- the fixed-panel endpoints do not equal the chain endpoints.

## Scientific boundary

L32 establishes a strong auditable substrate for repeated learning:

```text
real neural update
+ per-cycle old/new held-out court
+ exact checkpoint chain
+ direct initial-vs-final retention panel
```

It still does not prove unlimited lifelong learning.

A later production authority must add transactional interruption recovery,
atomic promotion/rollback and repeated real-data runs over longer calendar
periods before an autonomous updater can be trusted to modify the serving
checkpoint.
