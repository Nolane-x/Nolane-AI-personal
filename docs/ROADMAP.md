# Living Architecture Roadmap

## L0 — Persistent organism shell

- [x] identity survives restart
- [x] heartbeat advances state without an LLM call
- [x] append-only event timeline
- [x] snapshot + digest after every transition
- [x] rollback primitive
- [x] local episodic memory
- [x] bounded memory retrieval
- [x] unresolved conversation threads
- [x] initiative and silence decision
- [x] Qwen3-0.6B local cortex adapter
- [x] always-on CLI event loop
- [x] core replay/continuity tests

## L1 — Social observer

Add one structured model observer that proposes, but cannot directly commit:

- user affect hypotheses with confidence;
- memory candidates;
- thread updates;
- relationship evidence;
- conversational intent.

Validator requirements:

- bounded per-event deltas;
- source-event provenance;
- uncertainty preserved;
- reversible commit;
- inference never stored as fact without evidence.

## L2 — Recurrent living core

Train a small recurrent/state-space core on:
(previous_state, event_embedding, delta_t) -> proposed_next_state.

Matched court:

- deterministic reducer baseline;
- learned recurrent candidate;
- same replay histories;
- same state dimensionality;
- continuity, calibration, drift, initiative safety and compute measured.

## L3 — Persistent latent continuity

Checkpoint a compact neural latent state outside the Qwen KV cache and inject it into selected layers/adapters. Restart must not require replaying the full conversation.

## L4 — Rest/consolidation

Idle periods may propose:

- duplicate-memory merging;
- semantic preference extraction;
- working-state decay;
- unresolved-thread review;
- habit/personality updates.

Live conversations never update model weights directly.

## L5 — Architecture surgery

Only after replay benchmarks exist:

- insert recurrent/state-space blocks;
- add social-state heads;
- dynamic-depth routing;
- distill unused language capacity;
- compare against untouched Qwen3-0.6B under identical histories.

No surgery is promoted without measured benefit.
