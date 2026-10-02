# L39 Unified Continual Model Ledger

## Purpose

L31 made the factorized language boundary plastic.

L38 made the Nolane-owned recurrent cortex plastic.

Those two update paths have different neural ownership rules, so they must not
be forced through one old schema.

L39 creates a new ledger that understands both update types explicitly:

```text
L31 -> BOUNDARY_UPDATE
L38 -> CORTEX_UPDATE
```

The ledger tracks the whole model state rather than only one changed surface.

## Four continuity layers

For every adjacent cycle N -> N+1, L39 requires:

```text
artifact(N) == parent_artifact(N+1)

boundary_after(N) == boundary_before(N+1)

cortex_after(N) == cortex_before(N+1)

model_state_after(N) == model_state_before(N+1)
```

The composite model state is:

```text
SHA256({
  boundary_state_digest,
  cortex_state_digest
})
```

This prevents a valid artifact chain from hiding a neural-state discontinuity.

## L31 normalization

An L31 cycle changes the low-rank factorized language boundary.

L39 reuses the full existing L31 verifier, then normalizes:

- boundary before/after;
- unchanged cortex before/after;
- artifact parent/output SHA;
- L30 receipt;
- adaptation protocol lineage;
- composite model-state before/after.

L31 keeps its original authority and schema.

## L38 normalization

An L38 cycle changes the recurrent cortex while freezing the language boundary.

L39 independently verifies:

- L38 run schema and authority;
- L38 lineage self-digest;
- parent artifact binding;
- embedded L38 training receipt self-digest;
- L30 PASS;
- zero boundary gradients;
- actual cortex gradients;
- unchanged candidate boundary;
- changed candidate cortex;
- unchanged frozen reference;
- composite model-state before/after;
- L30 pre/post identity exactly equals those composite states;
- saved artifact boundary/cortex digests;
- saved training receipt;
- artifact dataset fingerprint equals lineage SHA.

L38 does not masquerade as L31.

## Cortex evidence is mandatory

The default policy is:

```text
min_cycles = 2
min_cortex_cycles = 1
require_unique_adaptation_protocols = true
```

Therefore an L31-only chain cannot earn a PASS from L39.

This makes the new ledger meaningful as evidence for recurrent-cortex
plasticity rather than a renamed copy of L32.

## Adaptation replay

Adaptation protocol SHA values must be unique across **all** update types.

For example:

```text
L31 uses protocol X
L38 reuses protocol X
```

is BLOCKED by default.

Changing the neural surface does not turn replayed evidence into a fresh
learning window.

## Long-horizon endpoint court

Per-cycle L30 PASS is still insufficient.

L39 uses one fixed held-out panel to compare the final checkpoint directly with
the checkpoint that existed before cycle 1.

Unlike the L32 boundary-only evaluator, the L39 endpoint evaluator permits both
boundary and cortex state to change.

It still requires:

- same L16 ancestry;
- same factorized boundary configuration;
- same recurrent cortex configuration;
- same recurrent-state carry policy;
- L28 PASS for the fixed panel;
- same exact held-out examples at both endpoints.

The existing L32 long-horizon court remains the metric receipt because its
mathematics already compares arbitrary checkpoint endpoints. L39 adds endpoint
boundary/cortex state digests to the receipt.

## Real endpoint evaluation

```bash
python scripts/evaluate_unified_long_horizon_retention.py \
  --initial /path/to/checkpoint-0/factorized-nolane.pt \
  --final /path/to/checkpoint-N/factorized-nolane.pt \
  --dataset /private/fixed-panel/personalization.jsonl \
  --protocol /private/fixed-panel/personalization-protocol-v1.json \
  --output runtime-data/l39/long-horizon-retention.json
```

Default limits remain:

- overall NLL regression <= +0.01;
- worst independent source-group regression <= +0.03.

## Unified chain verification

Receipts may be mixed in order:

```bash
python scripts/assess_unified_continual.py \
  --cycle runtime-data/cycle-1/l31-run-receipt.json \
  --cycle runtime-data/cycle-2/l38-run-receipt.json \
  --cycle runtime-data/cycle-3/l38-run-receipt.json \
  --long-horizon runtime-data/l39/long-horizon-retention.json \
  --output runtime-data/l39/unified-chain.json
```

The output records:

- number of L31 boundary cycles;
- number of L38 cortex cycles;
- every parent/output artifact SHA;
- every boundary before/after digest;
- every cortex before/after digest;
- every composite model-state before/after SHA;
- per-cycle L30 court SHA;
- per-cycle lineage SHA;
- moving-window retention/adaptation metrics;
- fixed long-horizon court SHA.

No raw prompts, targets or source-group identifiers are copied into the ledger.

## Fail-closed conditions

L39 blocks when:

- fewer than the configured minimum cycles exist;
- no recurrent-cortex cycle exists;
- an L31 or L38 receipt fails its native verifier;
- an embedded L30 receipt is missing, tampered or not PASS;
- artifact ancestry breaks;
- boundary continuity breaks;
- cortex continuity breaks;
- composite model-state continuity breaks;
- adaptation protocol is replayed;
- fixed endpoint court is BLOCKED;
- fixed endpoint artifact SHAs do not match the chain endpoints.

## CI courts

The L39 synthetic evidence court proves:

- exact L31 -> L38 continuity can PASS;
- cortex discontinuity blocks even when artifact ancestry is valid;
- composite model-state discontinuity is detected;
- L31-only chains cannot claim recurrent plasticity evidence;
- adaptation replay across L31/L38 is blocked;
- artifact discontinuity is blocked;
- L38 model-state tampering is detected;
- failed fixed-panel retention blocks;
- unified receipt tampering is detected;
- policy can require multiple recurrent-cortex cycles.

## Promotion boundary

L39 remains:

```text
UNIFIED_CONTINUAL_MODEL_CHAIN_NO_PRODUCTION_AUTHORITY
```

Existing L35 promotion authorization only understands the L32/L31 chain.

That is intentional.

The next authority wave must explicitly understand the L39 schema, recompute
the mixed evidence, and bind both boundary and cortex plasticity into promotion
authorization. L38 artifacts must not enter production by pretending to be
L31 cycles.
