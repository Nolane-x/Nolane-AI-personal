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

Engineering closure: **COMPLETE**.

- [x] structured `SocialProposal`
- [x] Qwen can reuse the already-loaded model as observer
- [x] proposal-only authority boundary
- [x] deterministic `MutationValidator`
- [x] bounded affect delta
- [x] bounded relationship delta
- [x] source-event provenance on memories
- [x] low-confidence facts downgraded to inference
- [x] bounded thread creation/resolution
- [x] observer uncertainty persisted explicitly
- [x] observer failure is fail-closed and audited
- [x] mutation receipt stored as its own event
- [x] CLI opt-in via `--social-observer`

L1 does not claim the observer infers human emotion perfectly. It gives model inference a constrained, reversible persistence boundary.

## L2 — Recurrent living core

Engineering substrate: **DEV-READY / UNPROMOTED**.

- [x] deterministic replay records from real runtime history
- [x] exact state vector contract
- [x] deterministic event featurizer
- [x] explicit time features
- [x] 32D recurrent latent state
- [x] tiny GRU transition model
- [x] bounded state-delta head
- [x] action-prior head
- [x] confidence head
- [x] exact analytical parameter audit
- [x] default core footprint: **14,451 parameters**
- [x] replay trainer with truncated BPTT
- [x] gradient clipping
- [x] checkpoint + manifest + DB/checkpoint SHA-256
- [x] replay court infrastructure
- [ ] train on a sufficiently large real interaction history
- [ ] freeze train/dev/test replay partitions
- [ ] beat deterministic baseline on held-out continuity/calibration gates
- [ ] promote recurrent core to production state authority

The neural core remains development-only until those final scientific gates pass.

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
