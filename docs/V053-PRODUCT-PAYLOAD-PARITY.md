# v0.53 Product Payload Parity

## Goal

v0.53 removes a subtle desktop/Android behavior split.

By v0.52, native Android could reproduce the pinned Qwen chat template exactly,
but the **dynamic user payload** was still assembled only inside the Python
ProductCortex.

That payload carries the parts that make Nolane personal:

- preferred name;
- language policy;
- response length;
- conversation style;
- bounded personal instruction;
- relationship state;
- affect/control state;
- unresolved open threads;
- requested intent;
- relevant memories;
- reply vs initiative task;
- current user text.

v0.53 freezes this dynamic payload as a structured contract and requires
desktop Python and native Rust to render it identically.

No new primary UI is added.

## Canonical structured payload

Python now creates:

```text
NOLANE-V053-PRODUCT-PAYLOAD-INPUT-V1
```

before rendering prompt text.

The schema contains only the fields that the existing product prompt already
used. It does not grant model, persistence or promotion authority.

Limits remain explicit:

- at most 4 unresolved open threads;
- at most 8 relevant memories;
- mode must be `reply` or `initiative`;
- initiative payloads must not contain user text.

Unknown mode/language/style/response-length values fail closed in native
rendering.

## Cross-language-safe formatting

The old desktop payload represented open threads using Python `repr(list)`.

That is not an appropriate native contract because quote/backslash escaping is
Python-specific.

v0.53 replaces that one field with compact UTF-8 JSON:

```text
open_threads=["Android local","Học toán","Tin AI 🚀"]
```

Python and Rust both render the exact same representation.

All other product wording intentionally remains the existing product wording.

## Desktop ownership

`FactorizedProductCortex` no longer carries a private copy of the profile,
state and memory prompt assembly logic.

Desktop generation now goes through:

```text
ProductProfile + CortexRequest
       |
       v
ProductPayloadInput
       |
       v
canonical product user payload
       |
       v
frozen chat-template contract
```

The desktop product is therefore using the same payload contract that Android
must satisfy.

## Native ownership

`nolane-mobile-runtime` now implements the same structured payload contract.

It owns deterministic rendering of:

- language guidance;
- conversational-style guidance;
- relationship/affect formatting;
- compact JSON open-thread list;
- memory bullet list;
- reply/initiative task wording.

The native runtime also derives the existing response-length token budgets:

- compact -> 96;
- balanced -> 160;
- expansive -> 256.

Sampling policy is deliberately not changed in this wave.

## Pinned-Qwen parity court

The exact pinned Qwen3-0.6B tokenizer fixture now carries three real product
payload cases:

1. Vietnamese direct reply with Unicode, quotes, backslashes, multiline memory,
   >4 threads and >8 memories;
2. English playful reply with a different response budget;
3. automatic-language natural initiative with no user text or memories.

For every row, CI requires Rust to match Python on:

- product system prompt;
- rendered dynamic user payload byte-for-byte;
- response-length token budget;
- complete frozen chat-template output byte-for-byte;
- exact Qwen token IDs token-for-token.

This court uses the same pinned Qwen tokenizer/prompt contract already used by
v0.51-v0.52.

## What v0.53 does not claim

v0.53 does not yet close end-to-end Android local chat.

Still open:

- seeded sampling parity with desktop;
- persistence/migration of Android latent/profile/runtime state;
- native local target wiring in the Tauri host;
- L36-authorized mobile release package binding;
- emulator/device local-chat court;
- resource/battery court.

Those remain explicit gates before Android can be called production-complete.
