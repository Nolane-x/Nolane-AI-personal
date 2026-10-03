# v0.56 Tauri LocalMobile Target

## Goal

v0.56 closes the Android app-routing split. The Tauri shell can now use the
native Rust Nolane runtime as its product backend instead of requiring a paired
HTTP runtime.

This wave does not claim production-complete Android local chat. Release
authority and device evidence remain separate gates.

## Product host

The native mobile runtime now contains a product-layer host above token
generation.

It exposes the product API shapes used by the existing Ember Quiet UI:

- `GET /v1/status`
- `GET /v1/readiness`
- `GET /v1/history`
- `GET /v1/profile`
- `PUT /v1/profile`
- `POST /v1/power`
- `POST /v1/chat`

A local turn executes the v0.55 persisted product state through the v0.54
seeded native generation path. Accepted turns advance interaction count and the
local product state version, and bounded conversation history is persisted.

Learning-review endpoints intentionally fail explicitly in v0.56. Android is
not allowed to pretend that the full desktop evidence-review/continual-learning
surface has been ported.

## First-launch continuity

The LocalMobile bundle contains a checkpoint-bound bootstrap state. On the
first local launch, the host creates a fresh device-local identity and writes
its own v0.55 persistent state into the app data directory.

Later launches reuse:

- identity;
- persistent latent;
- product profile;
- relationship/affect projection;
- open threads and bounded memories;
- interaction count;
- local state version;
- local conversation history;
- initiative and memory-enabled shell preferences.

Power state deliberately restarts as off.

## Bundle contract

Source-bundle schema:

`NOLANE-V056-LOCALMOBILE-BUNDLE-V1`

Expected resource layout:

```text
resources/mobile/
  localmobile-manifest.json
  bootstrap-state.json
  package/
  tokenizer.json
  tokenizer_config.json
  prompt-contract.json
```

The manifest binds the expected source checkpoint SHA-256 and prompt-contract
file SHA-256. The bootstrap state still passes the v0.55 internal integrity and
checkpoint courts.

The bundle manifest is not promotion authority. v0.57+ must bind these assets
to an L36-authorized release.

## Tauri routing

Android now has a dedicated `RuntimeTarget::LocalMobile`.

When valid bundled assets are present:

```text
web UI
  -> Tauri product_api
  -> LocalMobile Rust product host
  -> MobileRuntime
  -> native tokenizer / prompt / seeded generation
```

No loopback HTTP server, Python process or Transformers runtime is required in
this path.

Remote pairing remains available as an explicit development/fallback route.
Clearing that override returns Android to LocalMobile when the native host is
available.

The advanced remote controls are hidden while a local desktop or LocalMobile
target is active.

## Court

The deterministic Python mobile fixture now also emits a LocalMobile bundle
manifest and bootstrap state.

The Rust LocalMobile court must prove:

1. first-load native host creation;
2. unique device-local identity;
3. product-compatible status/profile;
4. power-on;
5. native local chat;
6. bounded history persistence;
7. profile mutation;
8. interaction/state-version advance;
9. restart continuity;
10. power resets to off;
11. unsupported learning routes fail explicitly.

The Android APK court separately compiles the Tauri application with the
Android-only path dependency on `nolane-mobile-runtime`.

## Explicit non-claims

v0.56 does not close:

- L36-authorized mobile release assets;
- clean emulator/device local-chat evidence;
- full desktop LivingEngine transition parity;
- mobile initiative/rest/learning parity;
- latency, RAM or battery courts.

Those remain required before Android local chat is called production-complete.
