# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.3.0

The runtime now contains two very different compute scales:

- **Qwen3-0.6B**: language cortex, used only when language inference is needed.
- **Tiny Living Core**: a recurrent 32D-latent model with only **14,515 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 has a **frozen held-out promotion court** and remains **UNPROMOTED** until real replay evidence passes it. L3 now has persistent neural latent continuity in **shadow-only mode** across restarts.

### Persistent runtime

- identity survives restart;
- heartbeat runs without invoking Qwen;
- SQLite event timeline and full state snapshots;
- SHA-256 state digests;
- auditable rollback;
- episodic memory and bounded retrieval;
- relationship and affect-control state;
- unresolved conversation threads;
- initiative where silence is a first-class action.

### L1 social observer

Qwen may optionally inspect a user message and propose structured updates. It never writes persistent state directly. Every proposal is clipped and validated by a deterministic authority boundary.

```bash
nolane-personal run --social-observer
```

Invalid observer output is recorded as a rejection and the existing state is preserved.

### L2 tiny recurrent core

The development core receives:

```text
(previous observed state, event features, delta-time, previous latent)
                              |
                              v
                    14,515-param GRU core
                       /       |        \
                      v        v         v
                next latent  state Δ  future-return
```

The future-return head predicts whether the user returns inside the configured horizon. Tail examples that do not expose the full future horizon are censored rather than treated as negatives.

Audit the neural core:

```bash
python -m pip install -e '.[neural]'
python scripts/audit_living_core.py
```

Freeze chronological replay evidence:

```bash
python scripts/freeze_replay_protocol.py \
  --db runtime-data/living.db \
  --output runtime-data/replay-protocol-v1.json
```

Train **only** on the frozen train split:

```bash
python scripts/train_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json
```

Evaluate on the frozen test split:

```bash
python scripts/evaluate_living_core.py \
  --db runtime-data/living.db \
  --protocol runtime-data/replay-protocol-v1.json \
  --checkpoint runtime-data/living-core-dev/living-core.pt
```

Default promotion requires protocol verification, checkpoint/protocol identity, enough held-out samples, state-fidelity gates, a future-return Brier improvement over the train base-rate baseline, and <=100K parameters.

See `docs/L2-PROMOTION-COURT.md` for the exact gates.

### L3 persistent latent shadow

The recurrent latent can now survive process restarts independently of Qwen context. It is atomically sealed with identity, exact checkpoint SHA-256, protocol SHA-256, latent dimension and the last processed state version.

Run one shadow catch-up:

```bash
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json
```

Continuously follow committed transitions:

```bash
python scripts/run_shadow_living_core.py \
  --db runtime-data/living.db \
  --checkpoint runtime-data/living-core-dev/living-core.pt \
  --protocol runtime-data/replay-protocol-v1.json \
  --follow
```

Shadow mode emits neural predictions and advances the latent but has **zero authority** over LivingState, memory, initiative or Qwen. See `docs/L3-PERSISTENT-LATENT.md`.

## Bootstrap Qwen

Model weights are intentionally **not committed to GitHub**. A pinned downloader reproduces the exact upstream checkpoint locally.

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

The model is downloaded into:

```text
models/Qwen3-0.6B/
```

## Install runtime

Core state/memory/heartbeat remains standard-library-only.

```bash
python -m pip install -e .
nolane-personal init
nolane-personal status
```

For local Qwen conversation:

```bash
python -m pip install -e '.[qwen]'
nolane-personal run
```

For the state/memory/heartbeat loop without loading any language model:

```bash
nolane-personal run --no-model --tick-seconds 5
```

## Pinned upstream

- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: Qwen3 causal LM
- License: Apache-2.0
- Role: initial language cortex, not the final architecture

## Direction

```text
events + time
     |
     v
persistent runtime
     |
     +--> validated social observer
     +--> local memory
     +--> initiative / silence
     |
     v
tiny recurrent living core
     |
     +--> held-out state fidelity
     +--> future-return prediction
     +--> frozen promotion court
     |
     v
persistent 32D latent (shadow)
     |
     +--> restart continuity
     +--> checkpoint/protocol binding
     +--> no production authority yet
     |
     v
future promoted adapters / wake policy
     |
     v
Qwen cortex
```

See `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/L2-PROMOTION-COURT.md`, and `docs/L3-PERSISTENT-LATENT.md`.
