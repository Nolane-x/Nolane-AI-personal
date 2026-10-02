# L29 Held-out Group Robustness Court

## Purpose

L28 ensures train/dev/test are structurally separated.

L29 addresses the next problem: a global held-out average can still hide a severe failure on one independent conversation group.

Example:

```text
group A regression = -0.08
group B regression = +0.10
global average      = +0.01
```

The average looks acceptable, but one held-out source group regressed badly.

L29 makes that failure visible and promotion-blocking.

## Court input

The court consumes:

- a verified personalization protocol;
- one held-out split, currently `test`;
- one per-example reference metric;
- one per-example candidate metric;
- a stage-specific worst-group non-inferiority threshold.

For L15/L17-L18/L19 the metric is weighted per-example NLL, where lower is better.

## Privacy

The protocol contains hashed source-group lineage.

The L29 receipt never copies those hash values.

Each group is represented only by a local alias:

```text
group_000
group_001
...
```

The receipt contains no raw prompt/target text and no raw or hashed conversation identifiers.

## Metrics

For every held-out source group L29 records:

- number of examples;
- split-local example indices;
- reference mean NLL;
- candidate mean NLL;
- candidate-minus-reference regression.

The summary records:

- overall reference mean;
- overall candidate mean;
- overall regression;
- mean group regression;
- worst group regression;
- best group regression;
- group-regression standard deviation;
- number of groups with positive regression.

The receipt is SHA-bound with `court_sha256`.

## Minimum independence

The real-candidate path now requires at least two independent held-out source groups.

Two held-out examples from the same conversation are not treated as two independent held-out groups.

Readiness therefore blocks before training with:

```text
insufficient_test_source_groups
```

if the frozen test split has fewer than two source groups.

## Stage integration

### L15 Native Boundary

Reference:

```text
L14 anchor-cortex candidate
```

Candidate:

```text
L15 native boundary
```

Worst-group threshold reuses:

```text
NativeQualityThresholds.max_degradation_vs_l14
```

### L17/L18 Factorized Boundary

Reference:

```text
L16 standalone dense boundary
```

Candidate:

```text
factorized boundary candidate
```

Worst-group threshold reuses:

```text
FactorizedQualityThresholds.max_nll_regression_vs_l16
```

### L19 Quantized Factor Runtime

Reference:

```text
source factorized candidate
```

Candidate:

```text
quantized factor candidate
```

Worst-group threshold reuses:

```text
QuantizedQualityThresholds.max_nll_regression_vs_factorized
```

No new arbitrary quality tolerance is introduced.

## Promotion hardening

L15/L17/L19 promotion scripts no longer trust the global quality decision by itself.

The effective quality status is accepted only when:

- an L29 receipt exists;
- its self-digest is valid;
- its status is `PASS`.

Missing, BLOCKED, or tampered L29 evidence makes the effective quality status empty, causing the existing promotion court to fail closed.

Promotion outputs also carry:

- group robustness status;
- group robustness court SHA-256.

## Courts

CI proves:

- a candidate can PASS when every held-out group remains within the stage threshold;
- a global +0.01 regression is BLOCKED when one group regresses +0.10;
- fewer than two independent held-out groups are BLOCKED;
- metric-length drift is rejected;
- receipt tampering is detected;
- raw prompt/target and source-group hashes are absent from receipts;
- promotion quality status disappears when the L29 receipt is missing, blocked, or tampered;
- real-candidate readiness blocks a two-example test split that comes from only one source group.

## Scientific boundary

L29 is a worst-group non-inferiority court.

It does not prove broad population generalization, and it does not perform cross-validation. Proper cross-validation would require retraining independent candidates per fold; L29 deliberately does not pretend that evaluating one already-trained candidate on alternate seen groups is valid cross-validation.

It strengthens the evidence we already have without inventing extra training claims.
