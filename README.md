# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language cortex is **Qwen3-0.6B**, but identity, time, memory, initiative and relationship continuity are owned by the Living Runtime rather than by prompt history.

## Current executable milestone: Living Runtime v0.1

The repository now includes:

- persistent identity that survives restart;
- event-driven heartbeat that runs without invoking Qwen;
- SQLite event timeline, full state snapshots and SHA-256 state digests;
- rollback without rewriting history;
- episodic memory with bounded retrieval;
- relationship and affect-control state;
- unresolved conversation threads;
- initiative scoring where silence is a first-class action;
- local Qwen3-0.6B cortex adapter;
- an always-on CLI loop;
- replay/continuity tests and GitHub CI.

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`.

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

Core state/memory/heartbeat has no runtime dependency outside Python's standard library.

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

To exercise only the living runtime without loading model weights:

```bash
nolane-personal run --no-model --tick-seconds 5
```

Inside the loop:

```text
/status
/thread <topic>
/quit
```

## Pinned upstream

- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: Qwen3 causal LM
- License: Apache-2.0
- Role: initial language/cognition scaffold, not the final architecture

## Architectural direction

Nolane AI Personal is being developed as:

```text
events + time
     |
     v
persistent living state
     |
     +--> local memory
     +--> initiative / silence
     +--> relationship continuity
     |
     v
Qwen language cortex
     |
     v
speech / action
```

The next research stages move progressively toward a learned recurrent/state-space living core, persistent latent continuity, dynamic computation and selective Transformer surgery. Those mechanisms will be promoted only after controlled replay tests show measurable benefit.

Qwen3-0.6B is the starting substrate. Nolane AI Personal is the system built around and eventually beyond it.
