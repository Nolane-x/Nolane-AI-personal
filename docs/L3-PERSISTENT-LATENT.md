# L3 Persistent Latent Shadow Mode

## Goal

Keep the Tiny Living Core's neural latent alive across process restarts without granting it production state authority before L2 promotion.

The latent is **not** the Qwen KV cache and does not require replaying the full conversation after every restart.

## Persistent latent identity

The latent checkpoint binds to:

- Living Runtime `identity_id`;
- exact neural checkpoint SHA-256;
- exact replay protocol SHA-256 when present;
- latent dimensionality;
- last processed LivingState version;
- monotonic latent sequence;
- a digest covering the complete latent record.

A mismatch is fail-closed. Changing the neural checkpoint cannot silently reuse an incompatible latent. Reset requires an explicit operator action.

## Shadow authority

Shadow mode may:

- read committed runtime transitions;
- advance the recurrent latent;
- predict bounded state deltas;
- emit action logits;
- estimate future-return probability;
- persist the next latent;
- write audit receipts.

Shadow mode may **not**:

- mutate LivingState;
- add/remove memories;
- open/resolve conversation threads;
- trigger Qwen;
- change initiative decisions;
- promote itself.

Every receipt declares:

`SHADOW_ONLY_NO_STATE_MUTATION`

## Run once

```bash
python -m pip install -e '.[neural]'
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json
```

## Follow new transitions

```bash
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json \
  --follow \
  --poll-seconds 5
```

The runner uses `source_state_version` as a database cursor. Runtime histories larger than 10,000 transitions do not force replay from version 1.

## Restart behavior

On restart, the runner loads the sealed latent and resumes after its cursor. If identity, checkpoint, protocol, latent dimension, or digest disagree, startup fails closed.

`--reset-latent` is the explicit escape hatch when an operator intentionally starts a new latent lineage.

## Scientific boundary

L3 shadow engineering can run before L2 promotion because it has zero production authority. Injection into Qwen adapters, wake control, or LivingState authority remains blocked until the relevant courts pass.
