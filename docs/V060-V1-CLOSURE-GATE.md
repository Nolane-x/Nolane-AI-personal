# v0.60 V1 Closure Gate

## Purpose

v0.60 is the software closure wave before v1.0.

It does not add another AI behavior layer. It turns the remaining release
requirements into one fail-closed evidence chain so the repository cannot call
itself v1.0 merely because source CI is green.

The closure result has only two meaningful outcomes:

- `READY_FOR_V1_0`
- `BLOCKED`

No synthetic CI receipt is allowed to substitute for the required real
longitudinal, promotion, clean-install, physical-device or performance evidence.

## Closure schema

`NOLANE-V060-V1-CLOSURE-RECEIPT-V1`

Authority:

`V1_RELEASE_READINESS_EVALUATION_NO_PROMOTION_AUTHORITY`

The closure evaluator never promotes a checkpoint. It only verifies that
already-existing evidence agrees on one release candidate.

## Required evidence

### 1. Real longitudinal learning

The evaluator requires a valid
`NOLANE-L43-REAL-LONGITUDINAL-REPORT-V1` with:

- status `PASS`;
- at least five real cycles;
- privacy fields closed;
- a final checkpoint SHA-256.

### 2. Final promotion ceremony

A COMPLETE L36 ceremony is required.

The L43 final checkpoint must be exactly the ceremony's promoted candidate.
Known synthetic promotion transaction IDs are blocked even when their mechanics
are otherwise valid.

### 3. Windows clean install

The v0.49 clean-install receipt must prove:

- installed model == promoted candidate;
- installed ceremony == selected L36 ceremony;
- expected product version;
- runtime readiness PASS;
- AI powered on;
- non-empty local reply;
- persisted history;
- installed desktop app spawned its bundled runtime.

### 4. Physical Android device

v0.60 freezes:

`NOLANE-V060-ANDROID-PHYSICAL-DEVICE-EVIDENCE-V1`

The receipt binds:

- exact APK SHA-256;
- exact candidate checkpoint;
- exact promotion ceremony;
- hashed device fingerprint;
- Android SDK;
- installed-version check;
- first boot;
- local chat;
- force-stop/restart;
- identity continuity;
- history continuity.

The raw device fingerprint, chat text and local data path are intentionally not
stored in the receipt.

`SYNTHETIC_COURT` receipts remain useful for unit tests but are rejected by
the v1 closure evaluator. Production closure requires
`REAL_PHYSICAL_DEVICE`.

## Android performance policy

Frozen policy:

`config/v1-android-performance-policy.json`

Initial v1 acceptance floor:

- minimum samples: 5;
- cold boot <= 30,000 ms;
- p95 local turn <= 30,000 ms;
- peak PSS <= 2,048 MiB;
- battery drain <= 25% / hour;
- maximum Android thermal status <= 3;
- crashes == 0.

The policy is deliberately a release floor, not a claim that these numbers are
ideal. Later releases may tighten them, but a single test run cannot silently
loosen its own threshold.

Performance evidence is bound to the exact device receipt, APK and checkpoint.

## Tooling

### Build physical-device evidence

```bash
python scripts/build_android_device_evidence.py \
  --apk /release/Nolane.apk \
  --ceremony /release/promotion-ceremony.json \
  --observation /private/device-observation.json \
  --product-version 1.0.0 \
  --output /private/android-device-receipt.json
```

The private observation may contain the raw device fingerprint. The output
receipt contains only its SHA-256.

### Build performance evidence

```bash
python scripts/build_android_performance_evidence.py \
  --device-receipt /private/android-device-receipt.json \
  --measurements /private/android-performance-measurements.json \
  --policy config/v1-android-performance-policy.json \
  --output /private/android-performance-receipt.json
```

### Evaluate v1 closure

```bash
python scripts/evaluate_v1_closure.py \
  --longitudinal-report /private/l43-report.json \
  --promotion-ceremony /release/promotion-ceremony.json \
  --windows-clean-install /private/windows-clean-install.json \
  --android-device /private/android-device-receipt.json \
  --android-performance /private/android-performance-receipt.json \
  --android-performance-policy config/v1-android-performance-policy.json \
  --expected-version 1.0.0 \
  --output /private/v1-closure.json
```

The command exits non-zero while the closure is BLOCKED.

## CI court

Ordinary CI verifies the mechanics only:

- missing evidence blocks;
- synthetic ceremony blocks;
- synthetic device evidence blocks;
- L43/L36 checkpoint mixing blocks;
- out-of-policy performance blocks;
- one fully consistent test fixture reaches the READY state mechanically.

That last unit fixture is not production evidence. The actual v1 release still
requires the real receipts listed above.

## What remains after v0.60 code closure

If L60 code/CI closes, the remaining work is evidence execution rather than a
new architecture wave:

1. run the real 5+ window L43 campaign;
2. promote that exact final checkpoint through real L40/L33/L34/L36;
3. run the real Windows release + clean-install court;
4. build the authority-bound Android release APK;
5. run physical-device local-chat/restart evidence;
6. collect device performance/battery/thermal measurements;
7. evaluate the closure receipt;
8. only if it says `READY_FOR_V1_0`, bump to v1.0.0.
