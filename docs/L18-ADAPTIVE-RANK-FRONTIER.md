# L18 Adaptive Rank Frontier

## Purpose

L17 proves that a low-rank language boundary can be trained and evaluated safely.

L18 removes the arbitrary choice of rank.

Instead of deciding in advance that rank 128, 96 or 64 is "small enough", L18 runs a descending frontier and accepts smaller ranks only while measured development evidence remains inside the frozen quality budget.

Default schedule:

```text
256 -> 192 -> 128 -> 96 -> 64
```

The schedule is a search order, not a promised final rank.

## Independent candidates

Every rank candidate is initialized from the same dense L16 language boundary.

This avoids a lossy chain such as:

```text
L16 -> rank256 -> rank192 -> rank128 -> ...
```

where errors from one compression stage would contaminate the next.

Instead:

```text
           +-> rank256 -> distill -> dev court
L16 dense -+-> rank192 -> distill -> dev court
           +-> rank128 -> distill -> dev court
           +-> rank96  -> distill -> dev court
           +-> rank64  -> distill -> dev court
```

The Nolane recurrent cortex is cloned exactly from L16 and remains frozen for every candidate.

## Per-rank gates

A candidate rank is accepted only if:

- it actually reduces boundary parameter count;
- cortex digest remains identical to L16;
- dev NLL regression versus L16 stays within budget;
- incremental regression versus the previously accepted rank stays within budget;
- greedy-token agreement versus L16 stays above threshold.

The first failed rank stops the frontier. The smallest previously accepted rank becomes the selected candidate.

Default development gates:

- max boundary parameter ratio: 0.60;
- max NLL regression per rank step: 0.02;
- max NLL regression versus L16: 0.05;
- minimum greedy-token agreement: 0.93.

These are development selection gates, not production promotion criteria.

## Search

```bash
python scripts/search_rank_frontier.py \
  --ranks 256,192,128,96,64
```

The command:

1. verifies the frozen personalization protocol;
2. loads the exact L16 source checkpoint;
3. encodes only train/dev evidence;
4. factorizes and distills each candidate independently;
5. stops at the first rejected smaller rank;
6. saves the smallest accepted factorized model;
7. writes an immutable rank-frontier receipt.

The receipt binds:

- exact source L16 checkpoint SHA-256;
- attempted ranks;
- accepted ranks;
- selected rank;
- selected checkpoint SHA-256;
- dataset fingerprint;
- per-stage reconstruction/quality/compression evidence.

## Held-out promotion

Rank selection never uses the frozen test split.

After selection, the chosen artifact still has to pass the L17 held-out quality and resource courts.

L18 promotion then checks that:

- held-out quality PASS;
- resource PASS;
- quality evidence rank equals the selected frontier rank;
- frontier, quality and resource evidence reference the same selected checkpoint;
- all three reference the same source L16 checkpoint.

```bash
python scripts/evaluate_factorized_boundary.py \
  --factorized runtime-data/l18-rank-frontier/factorized-nolane.pt

python scripts/benchmark_factorized_resources.py \
  --factorized runtime-data/l18-rank-frontier/factorized-nolane.pt

python scripts/decide_rank_frontier_promotion.py \
  --frontier runtime-data/l18-rank-frontier/rank-frontier-receipt.json \
  --quality runtime-data/l18-quality.json \
  --resources runtime-data/l18-resources.json
```

## Neural court evidence

CI verifies:

- rank schedule must be strictly descending;
- a permissive tiny-model frontier can progress 8 -> 4 -> 2;
- boundary parameter ratio falls at every accepted stage;
- a full-rank candidate that does not compress enough fails closed;
- source cortex digest remains unchanged;
- every selected candidate has an unchanged cortex;
- promotion fails when selected rank, selected checkpoint or source L16 lineage is mismatched.

## Scientific boundary

L18 chooses a low-rank operating point more rigorously, but it does not establish which rank is best for real Qwen3-0.6B-derived language weights.

That answer requires a real L16 checkpoint, sufficient approved Vietnamese/English and personal evidence, and the frozen held-out courts.

A native tokenizer/vocabulary remains a later question. It should not be introduced until the rank frontier shows how much of the inherited language boundary can already be removed without changing token semantics.
