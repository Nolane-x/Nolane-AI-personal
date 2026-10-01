# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity belong to the Living Runtime rather than to prompt history.

## Current executable milestone: Living Runtime v0.2

The runtime now contains two very different compute scales:

- **Qwen3-0.6B**: language cortex, used only when language inference is needed.
- **Tiny Living Core**: a recurrent 32D-latent transition model with only **14,451 parameters** by default.

L0 persistent runtime is complete. L1 validated social-observer engineering is complete. L2 recurrent-core infrastructure is executable but remains **DEV-READY / UNPROMOTED** until held-out replay evidence earns state authority.

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
                    14,451-param GRU core
                         /       |       \
                        v        v        v
                   next latent  delta   confidence
```

Audit the neural core:

```bash
python -m pip install -e '.[neural]'
python scripts/audit_living_core.py
```

Train a development checkpoint from the local runtime history:

```bash
python scripts/train_living_core.py --db runtime-data/living.db
```

Run the exact deterministic replay self-court:

```bash
python scripts/run_replay_court.py --db runtime-data/living.db
```

A trained neural core is **not** automatically promoted. Held-out replay partitions and promotion gates are still required.

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
tiny recurrent living core  <-- development authority court
     |
     +----------------------+
     |                      |
     v                      v
persistent latent       wake decision
                            |
                            v
                       Qwen cortex
                            |
                            v
                       speech/action
```

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for the promotion boundary and later L3-L5 architecture surgery.
