# Nolane AI Personal

**Nolane AI Personal** is an experimental always-on personal AI designed to persist through time instead of behaving as a stateless prompt-response chatbot.

The initial language model scaffold is **Qwen3-0.6B**. The model weights are intentionally **not committed to GitHub**. A pinned downloader reproduces the exact upstream checkpoint locally.

## Bootstrap model

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
```

The model is downloaded into:

```text
models/Qwen3-0.6B/
```

The directory is ignored by Git.

## Pinned upstream

- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Architecture: Qwen3 causal LM
- License: Apache-2.0
- Role in this project: initial language/cognition scaffold, not the final architecture

## Direction

The project aims to evolve beyond a conventional Transformer chatbot toward a persistent living architecture with:

- persistent identity and latent state
- long-term episodic and relational memory
- affective dynamics
- initiative and silence as first-class actions
- continuous time / heartbeat
- event-driven cognition
- memory consolidation during idle periods
- eventual recurrent/state-space and specialized social components

Qwen3-0.6B is the starting substrate. Nolane AI Personal is the system built around and eventually beyond it.
