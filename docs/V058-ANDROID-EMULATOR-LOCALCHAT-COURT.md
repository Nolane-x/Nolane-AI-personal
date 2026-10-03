# v0.58 Android Emulator Native-Boot + Persistent Local-Chat Court

## Goal

v0.58 moves the Android evidence boundary out of host-only Rust tests and into a
real packaged Tauri Android process.

The court installs an x86_64 APK into an Android emulator, launches the real app
package, lets Tauri load the authority-bound LocalMobile resources from the APK,
executes native local chat, force-stops the process, launches it again, and
requires identity/state/history continuity.

No user-facing feature is added in this wave.

## Why x86_64

The production Android release workflow remains arm64-first.

GitHub-hosted Android emulators are materially more reliable with x86_64/KVM, so
v0.58 builds a separate x86_64 court APK from the exact same deterministic
authority-bound mobile bundle used by native CI. The existing arm64 APK court
remains intact.

This changes the court architecture only, not the production architecture.

## Synthetic-only packaged probe

The Tauri process contains an Android-only court probe. It activates only when
both are true:

- promotion transaction ID is exactly
  `synthetic-mobile-release-court`;
- candidate checkpoint SHA-256 is the deterministic fixture digest
  `7777777777777777777777777777777777777777777777777777777777777777`.

A normal real L36 release does not satisfy this pair and therefore never runs
the automatic probe.

The probe uses the same `LocalMobileProductRuntime` instance exposed to the
normal Tauri `product_api` route.

## First boot

On a clean emulator install the probe requires:

- strict LocalMobile bundle load from packaged resources;
- fresh device-local identity;
- powered-off startup;
- state_version/history initially clean;
- successful native `POST /v1/power`;
- successful native `POST /v1/chat`;
- state_version increments by one;
- interaction count increments by one;
- history grows by exactly two rows.

It then stores a small court receipt in app data and logs:

`NOLANE_V058_FIRST_BOOT_PASS`

## Process restart

The emulator court then runs Android `am force-stop`, not a graceful in-app
shutdown.

On the next process launch the probe requires:

- same device-local identity;
- exact state_version from the first process;
- exact interaction count;
- exact history length;
- powered-off startup;
- another successful native local-chat turn.

It then logs:

`NOLANE_V058_RESTART_PASS`

Any probe failure logs:

`NOLANE_V058_EMULATOR_COURT_FAIL`

and aborts app setup.

## CI topology

1. Android native job builds the deterministic model/tokenizer/prompt/state.
2. The v0.57 staging path creates the authority-bound synthetic court bundle.
3. That bundle is uploaded as a CI artifact.
4. A separate x86_64 Tauri APK job embeds the bundle.
5. The APK is installed into an Android emulator.
6. The emulator court waits for first-boot PASS.
7. It force-stops the package.
8. It launches the package again and waits for restart PASS.

The court also verifies the installed app version is exactly `0.58.0`.

## Non-claims

This does not prove:

- that an APK built from real private COMPLETE L36 evidence passed;
- physical-device correctness;
- performance/thermal/battery fitness;
- full mobile LivingEngine initiative/rest/learning parity.

Those remain separate closure gates.
