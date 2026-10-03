# v0.59 Mobile Lifecycle Parity

## Goal

v0.59 moves LocalMobile beyond restart continuity into persistent event/time
behavior.

The Android-native product host now owns a lightweight lifecycle sidecar that
tracks the parts of the desktop LivingEngine state which were not present in
the frozen v0.55 product-state contract. The v0.55 neural/product state remains
unchanged, so existing identity, latent, profile and relationship state migrate
without destructive rewriting.

Rust-authored persistent state now uses
`NOLANE-V059-PERSISTENT-STATE-TYPED-INTEGRITY-V1`: integrity is computed over
a canonical projection in which neural f32 and behavioral f64 values are
represented by their IEEE bit patterns. This removes JSON decimal-spelling
drift across save/restart cycles. Legacy v0.55 canonical-JSON envelopes remain
readable, and Python implements the same typed projection so the bridge remains
bidirectional.

This wave targets **time dynamics + user-event dynamics + initiative +
conservative REST scheduling**. It does not claim full desktop memory
provenance or continual-learning parity.

## Lifecycle metadata

Schema:

`NOLANE-V059-LOCALMOBILE-LIFECYCLE-META-V1`

The sidecar persists:

- product state version;
- memory-enabled and initiative preferences;
- lifecycle tick count;
- last event, user event and AI speech timestamps;
- social drive;
- working curiosity;
- REST cycle count and last REST timestamp;
- last REST source/new-memory counts.

Existing `NOLANE-V056-LOCALMOBILE-META-V1` files are accepted and migrated
in memory with deterministic defaults. Unknown metadata schemas fail closed.

## Time dynamics

LocalMobile ports the desktop slow dynamics for every lifecycle advance:

- valence -> 0.0 with 6h half-life;
- energy -> 0.62 with 8h half-life;
- playfulness -> 0.45 with 10h half-life;
- irritation -> 0.0 with 45m half-life;
- concern -> 0.0 with 4h half-life;
- social drive rises conservatively over a 6h scale.

Clock movement is monotonic and a single call is capped to seven days of
relaxation, matching the desktop runtime's bounded time advance.

## User-event dynamics

Before native reply generation, a real user message now updates the same
high-level controls as desktop:

- interaction count;
- familiarity and closeness;
- social-drive suppression;
- curiosity rise;
- energy rise;
- Vietnamese/English negative/positive/anger cue effects.

The updated state is the state seen by the native cortex for that reply. This
matches the desktop causal ordering: user event first, cortex generation second.

## Initiative

`POST /v1/tick` now exists on LocalMobile.

The mobile initiative baseline ports the desktop policy constants:

- 15 minute minimum user silence;
- 30 minute AI speech cooldown;
- 24 hour hard silence suppression when no thread is open;
- score threshold 0.66;
- concern/social-drive/curiosity/closeness components.

The frozen v0.55 state carries thread topics but not thread importance. Until a
richer frozen state contract exists, mobile treats every carried thread with
the desktop default importance of 0.5 and caps the thread component at the same
0.34 ceiling. This limitation is explicit rather than silently inventing
importance values.

If initiative fires, generation uses the normal product prompt, seeded
sampler, native kernel and persistent state path.

Android Tauri also owns a process-resident native heartbeat, matching the
desktop product server cadence:

- initiative off: 5 seconds;
- gentle: 30 seconds;
- active: 12 seconds.

The heartbeat is implemented in the Rust host rather than the web UI, so focus
changes do not stop lifecycle progression while the Android process remains
resident. The synthetic v0.58 packaged-emulator probe disables this heartbeat
to keep its frozen restart receipt deterministic.

## REST baseline

The 30 minute idle threshold and 45 minute cycle cooldown match the desktop
RestScheduler.

Mobile memory is currently a bounded string projection, not the desktop
provenance-rich MemoryRecord graph. Therefore v0.59 does **not** fabricate
ConsolidationReceipt parity. The mobile REST baseline only performs
conservative near-duplicate compaction with Jaccard lexical similarity >= 0.72
and records cycle/source counts.

Full provenance-rich consolidation remains a later gate.

## Court

The LocalMobile lifecycle court proves:

1. v0.56 metadata migrates to v0.59 without losing product settings;
2. user chat updates lifecycle state before native generation;
3. a tick five minutes after the user remains silent with
   `user_recently_active`;
4. REST is not run before the 30 minute idle window;
5. after 31 minutes, a high-concern/open-thread fixture reaches the same 0.66
   initiative threshold;
6. REST runs and compacts duplicate memory conservatively;
7. initiative uses the follow-up intent path;
8. lifecycle metadata survives a full product-host restart;
9. immediate subsequent REST is blocked by the 45 minute cooldown.

Synthetic court generation remains token-bounded. Production chat and
production initiative generation keep the normal product response-length
limits.

## Explicit non-claims

v0.59 does not yet prove:

- provenance-rich mobile consolidation identical to desktop;
- mobile social-observer mutation parity;
- continual-learning/review parity;
- a real COMPLETE-L36 physical-device campaign;
- final RAM/thermal/battery acceptance.

Those remain release gates rather than being hidden behind a broad "parity"
claim.
