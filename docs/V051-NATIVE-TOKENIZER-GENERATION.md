# v0.51 Native Tokenizer & Generation Host

## Goal

v0.51 turns the v0.50 one-token Rust neural kernel into a native text-to-text
generation substrate without adding any new permanent product UI.

The existing Ember Quiet chat surface remains the intended product interface.

## Native tokenizer

The runtime loads a local Hugging Face `tokenizer.json` directly in Rust.

It does not:

- download tokenizer assets;
- require Python;
- require Transformers;
- require a Qwen model object.

The Rust `tokenizers` dependency is built with default features disabled so
Android does not inherit the default progressbar/onig/C++ esaxx stack merely to
read and execute an already frozen tokenizer.

## Native prompt prefill

The runtime:

1. encodes text with the frozen tokenizer;
2. initializes Nolane recurrent state from the supplied persistent latent;
3. scans every prompt token through the v0.50 `MobileKernel`;
4. retains the final logits/state for autoregressive generation.

An empty tokenized prompt is rejected.

The native runtime bounds work to:

- at most 8192 prompt tokens;
- at most 512 generated tokens per call.

These are product-safety bounds, not architecture limits.

## Deterministic generation court

v0.51 starts with greedy generation deliberately.

For each generated token:

1. choose argmax of finite logits;
2. append the token;
3. stop immediately if it is EOS;
4. otherwise run that token through the recurrent kernel to obtain the next
   logits/state.

This exactly mirrors the deterministic branch of the existing standalone Nolane
generation path.

Sampling is intentionally deferred until the deterministic trajectory is
closed first.

## Cross-language tokenizer + generation evidence

The CI fixture now creates a real `tokenizer.json`, encodes a deterministic
prompt in Python, then executes the mobile graph to freeze:

- prompt token IDs;
- generated token IDs;
- EOS decision;
- final recurrent state;
- decoded generated text.

The Rust runtime must load the same tokenizer/package and reproduce all of those
outputs.

The native runtime crate must also compile for `aarch64-linux-android`.

## What remains open

v0.51 is not yet the full Android product runtime.

Still open:

- exact product/Qwen chat-template rendering;
- sampled generation with deterministic seeded courts;
- persistent product latent/profile/state bridge;
- Tauri `LocalMobile` runtime target;
- promoted mobile package + tokenizer Android resources;
- emulator/device end-to-end chat;
- latency, memory and battery courts.

## v1 product rule

Native local inference replaces backend plumbing beneath the existing chat
surface.

It must not create a second model dashboard, a second composer, or a separate
"mobile AI" product experience.
