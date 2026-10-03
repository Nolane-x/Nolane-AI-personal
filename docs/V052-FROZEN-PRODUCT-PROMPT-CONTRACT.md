# v0.52 Frozen Product Prompt Contract

## Goal

v0.52 closes the product chat-template boundary for Android without adding a
Jinja runtime, Python runtime or new UI to the APK.

The native inference path must consume the same rendered system+user prompt as
the desktop product path before local Android product wiring begins.

## Why freeze instead of reimplement Jinja

The pinned tokenizer owns the upstream chat template. Reimplementing that
template manually in Rust would duplicate an evolving Jinja program and make
silent prompt drift likely.

v0.52 freezes only the product subset that Nolane actually uses:

- one system message;
- one user message;
- generation prompt enabled;
- thinking disabled.

At release/export time Python Transformers renders two unique sentinel strings
through the exact tokenizer chat template.

The resulting prompt is split into:

- prefix before system content;
- separator between system and user content;
- suffix after user content.

Runtime rendering is then deterministic concatenation:

```text
prefix + system + between + user + suffix
```

There is no Jinja interpreter on Android.

## Integrity binding

The frozen contract binds:

- exact `tokenizer.json` SHA-256;
- exact `tokenizer_config.json` SHA-256;
- role sequence;
- generation/thinking flags;
- frozen segments;
- a sentinel probe render hash;
- a self-digested Python contract;
- the complete contract file SHA-256 consumed by Rust.

Rust refuses to load the contract if:

- expected contract-file SHA differs;
- tokenizer.json changed;
- tokenizer_config.json changed;
- schema/authority/roles/flags drift;
- the sentinel probe no longer reconstructs.

The contract grants no model or promotion authority.

## Synthetic end-to-end court

The deterministic mobile fixture now includes a frozen prompt contract.

The synthetic contract renders:

```text
system + " " + user
```

into the exact prompt already used by the v0.51 generation golden.

Rust must:

1. load the mobile model package;
2. load the tokenizer;
3. verify the prompt contract and tokenizer hashes;
4. render the system/user prompt;
5. produce the exact expected prompt token IDs;
6. run greedy native generation;
7. reproduce generated token IDs/text;
8. reproduce final recurrent state.

This proves the prompt contract is not merely a string helper; it preserves the
neural generation trajectory.

## Exact pinned Qwen court

The product court reads `model.lock.json` and resolves:

- repo: `Qwen/Qwen3-0.6B`;
- exact pinned revision from the lock file.

Only tokenizer assets are downloaded.

Python Transformers is the template-rendering authority for the court.
For multiple Vietnamese/English/Unicode/code-like system+user pairs:

1. Python `apply_chat_template(..., add_generation_prompt=True,
   enable_thinking=False)` freezes the expected prompt;
2. the frozen v0.52 contract must reproduce it byte-for-byte;
3. desktop AutoTokenizer IDs must equal the raw tokenizer boundary IDs;
4. Rust loads the same contract and tokenizer;
5. Rust rendered prompt must equal Python exactly;
6. Rust token IDs must equal the desktop token IDs exactly.

No 0.6B model weights are downloaded by this court.

## Product boundary

v0.52 still does not claim end-to-end local Android product chat.

Still open:

- native construction of Nolane's dynamic personalization/state/memory user
  payload;
- sampled generation parity with the desktop product;
- persistent latent/profile/state storage on Android;
- Tauri LocalMobile request routing;
- L36-authorized mobile model/prompt/tokenizer release staging;
- emulator/device local-chat court;
- latency/memory/battery evidence.

## v1 rule

The existing Ember Quiet chat surface remains unchanged.

Prompt parity is backend plumbing. It must not create a model selector, prompt
editor or separate Android AI screen.
