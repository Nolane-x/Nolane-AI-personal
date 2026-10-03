# L46 Real Learning Campaign

L46 is a backend-only campaign builder for the real longitudinal-learning path.

It deliberately adds no new primary product UI.

## Minimal product surface

The v1 product remains centered on:

- chat;
- AI power;
- personalization;
- explicit one-by-one learning review/correction.

L46 runs behind those surfaces. Its job is to turn finalized reviewed windows into a clean L43 campaign.

## Required campaign

A real campaign requires, in chronological order:

1. one fixed held-out window;
2. one baseline retention window;
3. at least five adaptation windows.

All windows must come from the same local learning registry and must already be finalized through L25/L27/L28.

## Critical anti-leakage rule

A prior adaptation window's original held-out test examples must never be rehearsed later.

For cycle N > 1, L46 composes retention only from:

- baseline train + dev;
- prior adaptation train + dev.

Original test splits remain protected for L43 learned-window retention measurement.

This prevents the experiment from "remembering" a skill merely because its old test answers were fed back into training.

## Cross-window isolation

L46 checks:

- exact prompt overlap;
- exact prompt/target pair overlap;
- near-duplicate overlap using token Jaccard + sequence similarity;
- chronological window order;
- role reuse;
- shared registry lineage.

The campaign receipt contains aggregate digests/counts, not raw prompt/target text.

## Build

```bash
python scripts/real_learning_campaign.py build \
  --spec /private/evidence/campaign-spec.json \
  --output-dir /private/evidence/l46-campaign
```

Then verify:

```bash
python scripts/real_learning_campaign.py verify \
  --output-dir /private/evidence/l46-campaign
```

The output includes one native L43 plan.

L46 does not execute training, issue L40 authority, or promote a checkpoint.

## v1 principle

L46 exists to make the strong feature reliable, not to add another feature.

The v1 UI is intentionally feature-frozen unless a new surface is required to complete a core user job that cannot be done safely through the existing chat/personalization/review flow.
