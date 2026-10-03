# v0.55 Persistent Mobile State Bridge

## Goal

v0.55 closes the state-continuity gap between the desktop Living Runtime and the
native Android inference host without adding a new primary UI.

The native runtime must no longer require an ad-hoc latent vector and separately
constructed product payload on every launch. One persisted state artifact now
binds the mobile cortex to the same identity/profile/relationship/affect/open
threads/memories and latent that the product runtime uses.

## Frozen state contract

Schema:

`NOLANE-V055-MOBILE-PERSISTENT-STATE-V1`

The envelope contains:

- schema;
- SHA-256 over the canonical semantic state object;
- source checkpoint SHA-256;
- persistent latent vector;
- product profile projection;
- identity + relationship + affect state;
- up to four unresolved open threads;
- up to eight relevant memories.

The state artifact is local user data. It carries no release or promotion
authority.

## Integrity and failure policy

Both Python and Rust:

- require a lowercase 64-hex checkpoint digest;
- reject checkpoint mismatch;
- reject latent shape mismatch when an expected dimension is known;
- reject non-finite latent/state values;
- reject empty identity;
- reject negative interaction counts;
- enforce the v0.53 open-thread/memory bounds;
- reject state files above 4 MiB;
- verify a canonical state SHA-256 before accepting state.

Writes use a same-directory temporary file, fsync, and atomic replacement.
A partial/corrupt/tampered state is never silently converted into a fresh
identity.

Rust verifies the integrity digest on the semantic JSON value before narrowing
latent values to f32. This keeps Python-authored files stable across the
cross-language boundary.

## Native runtime bridge

`MobileRuntime` now supports:

- loading directly from persistent state;
- loading prompt contract + persistent state together;
- attaching/replacing a validated state;
- saving the attached state;
- constructing the v0.53 product payload from persisted state;
- seeded product generation directly from that persisted state.

Existing v0.54 constructors remain valid so older courts do not change
semantics.

## Cross-language court

The deterministic mobile fixture now emits `persistent-state.json` from the
Python product/Living-State projection.

The Rust `verify_persistent_state_fixture` court must:

1. accept that exact Python-authored artifact;
2. bind it to the exact mobile checkpoint;
3. attach it to the native runtime with the frozen prompt contract;
4. reconstruct the product payload;
5. execute seeded native generation;
6. save the state from Rust;
7. reload it with identical semantic state.

This proves the bridge is not merely a Rust-only storage helper.

## Explicit non-claims

v0.55 does **not** yet claim Android product completion.

Still open:

- Tauri `LocalMobile` request routing;
- app-data path ownership/migration policy in the Android shell;
- L36-authorized mobile release asset binding;
- emulator/device end-to-end local chat;
- latency/memory/battery courts.

Those remain later release gates.
