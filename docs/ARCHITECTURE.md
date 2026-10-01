# Nolane AI Personal — Living Runtime v0.1

## Core thesis

The system is not modeled as prompt -> response -> reset.

State evolves continuously:

S(t+1) = F(S(t), event, delta_t)

Qwen3-0.6B is a replaceable language cortex. Identity, time, memory, initiative, relationship history and continuity belong to the runtime.

## Runtime layers

1. Event reducer receives user, clock and internal events.
2. Persistent LivingState carries affect-control state, relationship state, working state and unresolved threads.
3. SQLite stores events, full snapshots, SHA-256 state digests and episodic memories.
4. InitiativeEngine can choose speech or silence without requiring a fresh user prompt.
5. QwenCortex receives only compact state plus bounded relevant memory and realizes language.

The heartbeat does not require Qwen. Qwen wakes only when language generation is needed.

## What is inherited from Nolane-AIv2

Nolane-AIv2 already established useful architectural discipline:

- explicit state instead of hiding all cognition in token context;
- recurrent state transitions as a real mechanism;
- scoped memory with explicit insertion/query behavior;
- world/state separation;
- immutable identities and digests for evidence;
- mechanisms do not earn authority merely because code exists;
- negative results may remove complexity rather than being tuned away.

Nolane AI Personal carries those principles into a social, persistent runtime. It does not copy AIv2's mathematical task machinery.

## Persistent state

The v0.1 state contains a stable identity_id, monotonic snapshot version, affect-control variables, relationship variables, recent topics, curiosity, uncertainty, open threads and event timestamps.

Affect fields are behavioral control state. They are not a claim that the program is conscious or literally feels emotions.

## Memory and lineage

Private runtime data stays local under runtime-data/.

Every committed transition stores:

- the triggering event;
- the resulting full state snapshot;
- a canonical SHA-256 digest;
- optional episodic memories linked to the source event.

Rollback creates a new auditable transition rather than rewriting history.

## Initiative

Initiative is bounded by:

- unresolved threads;
- concern;
- social drive;
- curiosity;
- relationship closeness;
- minimum user-silence gate;
- speech cooldown;
- long-silence suppression when no thread remains open.

Silence is a first-class action.

## Architecture waves

L0: persistent explicit state + event loop + memory + initiative + Qwen cortex.

L1: structured social observer proposals. Model output proposes affect/memory/thread updates; a deterministic validator bounds and records changes.

L2: learned recurrent/state-space living core consuming previous state, event embedding and delta_t. It must beat the deterministic reducer under frozen replay histories before promotion.

L3: persistent latent identity vector checkpointed independently of token context and injected through adapters/gates into Qwen.

L4: dynamic computation: tiny living core handles simple events; deeper Qwen compute wakes only when needed.

L5: model surgery: replace selected Transformer blocks only when matched tests show continuity, quality, latency or compute gains.

The goal is not to be non-Transformer for branding. Architectural surgery survives only when it earns its complexity.
