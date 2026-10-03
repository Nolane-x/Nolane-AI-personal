# v0.54 Seeded Sampling Parity

## Goal

v0.54 removes the remaining stochastic decoding split between the desktop
product path and the native Android runtime without adding any new primary UI.

Given the same:

- model package;
- tokenizer and frozen prompt contract;
- structured product payload;
- persistent latent;
- sampling seed;
- temperature;
- top-p;

desktop Python and native Rust must produce the same generated token IDs,
decoded text and final recurrent state.

## Frozen sampler

The sampler schema is:

`NOLANE-V054-SEEDED-Q32-NUCLEUS-V1`.

It uses:

- SplitMix64 for one explicit portable RNG stream;
- logits quantized to millilogit integers;
- bounded quantized-logit range;
- Q40 exponential weights;
- Q32 probability/top-p threshold;
- deterministic weight-descending, token-id-ascending tie breaking;
- one RNG advance per sampled token.

The implementation rejects:

- empty logits;
- NaN/Inf logits;
- temperature <= 0 or non-finite;
- top-p outside (0, 1];
- finite logits outside the frozen quantization range.

## Minimal-nucleus semantics

Top-p retains the **smallest ranked token set whose cumulative quantized mass
reaches the requested Q32 threshold**.

The threshold uses ceil, not floor, so the retained set never represents less
than the requested quantized top-p mass.

This is frozen by equal-logit boundary tests:

- top_p=0.25 over four equal logits retains only token 0;
- top_p=0.50 can sample only tokens 0 or 1;
- top_p=1.00 exposes the full four-token set;
- a tiny positive top-p still retains exactly the highest-ranked non-zero token.

## RNG contract

SplitMix64 is frozen by known-answer vectors, not merely repeatability tests.

The seed is a generation input.

The normal desktop product may still obtain a fresh cryptographic 64-bit seed
for each response. The explicit seeded path exists so desktop/native parity,
bug reproduction and release courts can replay one exact stochastic trajectory.

## Desktop integration

`StandaloneNolaneLM.generate(..., sampling_seed=...)` uses the frozen seeded
sampler only when:

- `do_sample=True`; and
- an explicit sampling seed is provided.

`FactorizedProductCortex` now uses the seeded path for normal product
generation with the existing product policy:

- temperature = 0.78;
- top_p = 0.90.

A caller may inject a deterministic seed source for tests/courts; production
defaults to a fresh 64-bit cryptographic seed.

## Native integration

The Rust mobile runtime implements the same sampler and exposes seeded
generation beneath the existing product payload + frozen prompt contract.

The deterministic fixture freezes:

- prompt token IDs;
- generated sampled token IDs;
- decoded sampled text;
- EOS behavior;
- final recurrent state.

Rust must reproduce all of them.

The same native crate must continue to compile for
`aarch64-linux-android`.

## Numerical boundary

The cross-language court proves the frozen product trajectory on CI and the
sampler uses quantization to reduce sensitivity to tiny floating-point drift.

It does not claim every platform's libm `exp` is bit-identical for every
possible adversarial logit vector. Android emulator/device courts remain the
final authority before Android local chat is called production-complete.

## What remains after v0.54

The major Android blockers become product-state/runtime integration rather than
language-generation parity:

- persist/bind Android latent and local product state;
- wire the native runtime into Tauri as `LocalMobile`;
- bundle one L36-authorized mobile package + tokenizer + prompt contract;
- run emulator/device local-chat, latency, memory and battery courts.

No separate Android dashboard or model picker is planned.
