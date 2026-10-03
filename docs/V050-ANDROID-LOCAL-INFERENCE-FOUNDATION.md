# v0.50 Android Local Inference Foundation

## Goal

v0.50 removes the largest architectural blocker to running Nolane locally on
Android: the neural token step can now leave Python/PyTorch and execute as a
small deterministic Rust kernel.

This wave adds no new primary UI.

It is a foundation, not a claim that Android chat is production-complete.

## One-token contract

The mobile neural graph is deliberately split from the autoregressive host.

Inputs:

- one token id;
- packed recurrent state;
- persistent latent vector.

Outputs:

- next-token logits;
- next packed recurrent state.

The graph owns the factorized boundary and recurrent cortex math.

The native host will later own:

- prompt tokenization;
- token iteration;
- EOS handling;
- sampling;
- decoded text;
- product profile/state orchestration.

This keeps the neural kernel small and independently testable.

## Native equivalence before export

The Python mobile module is not an approximation.

Golden tests require it to match the existing
`StandaloneNolaneLM.forward(..., state=...)` path across multiple sequential
tokens for both tied and untied factorized boundaries.

The court compares:

- initial recurrent state;
- logits on every token;
- packed recurrent state on every token.

The current tolerance is 1e-6 for Python/native-model equivalence.

The public contract includes every normalization epsilon used by the graph:

- hidden LayerNorm epsilon;
- latent RMSNorm epsilon;
- final boundary RMSNorm epsilon.

A native implementation must not guess these constants.

## Python-free mobile package

`export_mobile_factorized_package()` emits:

```text
manifest.json
contract.json
weights.safetensors
```

The package:

- binds the exact source checkpoint SHA-256;
- converts floating weights to portable float32;
- excludes user latent state;
- excludes tokenizer assets;
- excludes chat text;
- carries no promotion authority;
- records tensor names/counts and file hashes;
- fails verification on contract/weights tamper.

Authority remains external. A mobile package cannot promote itself.

## Pure Rust kernel

`crates/nolane-mobile-kernel` implements the one-token graph directly in Rust:

- factorized embedding;
- LayerNorm/RMSNorm;
- fast/slow recurrent state;
- virtual-depth recurrence;
- bounded residual gate;
- factorized output projection.

It does not depend on Python, PyTorch, Candle, ONNX Runtime or a Qwen model
object.

The only model-file parser is the Rust safetensors crate.

## Cross-language court

CI generates one deterministic package and golden trajectory from Python.

Rust then loads that exact package and reproduces:

1. initial state;
2. five sequential token steps;
3. logits and recurrent state after each step.

The Rust/Python court uses a small floating-point tolerance and fails on any
trajectory drift.

The same crate must also `cargo check` for `aarch64-linux-android`.

## What v0.50 does not claim

v0.50 does not yet provide end-to-end local Android chat.

Still open:

- native tokenizer loading;
- exact product prompt/chat-template construction;
- autoregressive sampling loop;
- EOS/stop semantics;
- persistent latent/profile/state bridge;
- packaged promoted mobile weights;
- Tauri Android runtime target wiring;
- emulator/device latency and memory courts;
- full local chat smoke on Android.

These remain explicit follow-on gates.

## v1 principle

The Android path reuses the existing Ember Quiet chat UI.

Local inference should replace the current unconfigured/remote backend beneath
the same controls, not add a second dashboard or model-management surface.
