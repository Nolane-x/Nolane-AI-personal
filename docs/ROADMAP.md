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

Engineering substrate: **COURT-READY / UNPROMOTED**.

- [x] deterministic replay records from real runtime history
- [x] exact state vector contract
- [x] deterministic event featurizer
- [x] explicit time features
- [x] 32D recurrent latent state
- [x] tiny GRU transition model
- [x] bounded state-delta head
- [x] action-prior head
- [x] future-return prediction head
- [x] exact analytical parameter audit
- [x] default core footprint: **14,515 parameters**
- [x] replay trainer with truncated BPTT
- [x] gradient clipping
- [x] checkpoint + manifest + DB/checkpoint SHA-256
- [x] chronological train/dev/test freeze protocol
- [x] per-record before/after state digests
- [x] protocol self-digest and drift verification
- [x] censored future-return labels to prevent tail leakage
- [x] held-out evaluator
- [x] state-fidelity gates
- [x] Brier improvement gate against train base-rate
- [x] parameter-cap gate
- [x] checkpoint/protocol SHA match gate
- [ ] train on a sufficiently large real interaction history
- [ ] obtain >=50 held-out state cases and >=20 uncensored return cases
- [ ] pass the frozen promotion court
- [ ] promote recurrent core to production state authority

A neural core that only imitates the deterministic reducer has not earned replacement authority. It must preserve state continuity and add held-out predictive value.

## L3 — Persistent latent continuity

Engineering substrate: **SHADOW-READY / NO PRODUCTION AUTHORITY**.

- [x] 32D latent checkpoint independent from Qwen KV cache
- [x] atomic latent persistence across restart
- [x] latent self-digest verification
- [x] bind latent to Living Runtime identity
- [x] bind latent to exact neural checkpoint SHA-256
- [x] bind latent to replay protocol SHA-256 when present
- [x] persist source-state version cursor
- [x] scalable replay resume via `after_version`
- [x] shadow prediction receipts
- [x] explicit `SHADOW_ONLY_NO_STATE_MUTATION` authority
- [x] explicit reset required for incompatible latent lineage
- [ ] pass L2 promotion court on real replay history
- [ ] evaluate latent stability across long restarts
- [ ] inject promoted latent into selected Qwen adapters
- [ ] permit promoted latent to influence wake policy

L3 may run in shadow before L2 promotion because it cannot mutate state, memory, initiative or Qwen behavior.

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
