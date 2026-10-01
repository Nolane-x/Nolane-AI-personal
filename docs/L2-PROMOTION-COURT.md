# L2 Frozen Replay & Promotion Court

## Purpose

The Tiny Living Core is not allowed to become state authority because a training loss looks good.

Promotion requires a frozen chronological replay protocol and held-out evidence.

## Freeze

```bash
python scripts/freeze_replay_protocol.py \
  --db runtime-data/living.db \
  --output runtime-data/replay-protocol-v1.json
```

The protocol freezes:

- identity;
- version ordering;
- event IDs and event kinds;
- before-state digests;
- after-state digests;
- chronological train/dev/test membership;
- one protocol SHA-256.

New events may be appended after the freeze. Frozen history may not drift.

## Train

```bash
python -m pip install -e '.[neural]'
python scripts/train_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json
```

Training receives only the frozen train split.

The recurrent core learns two supervised objectives:

1. bounded next-state delta fidelity;
2. probability that the user returns within the configured future horizon.

Future-return examples whose complete horizon is not observable inside the split are censored instead of being labeled negative.

## Evaluate

```bash
python scripts/evaluate_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json \
  --checkpoint runtime-data/living-core-dev/living-core.pt
```

Evaluation carries latent state through the frozen chronological history but scores only the held-out test split.

## Default promotion gates

Promotion is blocked unless all gates pass:

- replay protocol verifies;
- checkpoint protocol SHA matches;
- at least 50 state-transition test cases;
- at least 20 uncensored future-return test cases;
- state MAE <= 0.030;
- maximum state error <= 0.200;
- future-return Brier score improves over the train base-rate baseline by at least 0.020;
- candidate stays <= 100,000 parameters.

A neural core that merely imitates deterministic state transitions has not earned replacement authority. It must preserve continuity **and** add predictive value unavailable to the simple baseline.

The current 14,515-parameter core remains `DEVELOPMENT_UNPROMOTED` until this court passes on real replay history.
