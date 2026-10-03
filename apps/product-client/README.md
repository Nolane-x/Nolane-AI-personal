# Nolane Product Client

This directory contains the deliberately small Windows + Android product surface for Nolane AI Personal.

## Product shape

The normal chat screen permanently exposes only:

- Nolane identity;
- one explicit AI power control;
- the transcript;
- the composer.

Deep personalization is available through one secondary sheet instead of permanent dashboard chrome.

The selected NUI direction is **Ember Quiet**: warm neutral surfaces, document-like assistant responses, compact user messages, and orange reserved for living/primary state.

## Architecture

```text
Windows
Tauri WebView
   |
   | invoke()
   v
Rust native host
   |
   | authenticated random loopback proxy
   v
PyInstaller product runtime
   |
   v
LivingEngine + FactorizedProductCortex
   |
   +-- local SQLite identity/memory/history
   +-- approved factorized-nolane.pt
   +-- COMPLETE L36 promotion ceremony
   +-- local tokenizer assets
```

The web layer never receives the loopback authentication token.

On Android, the same Tauri/web UI compiles as an APK. Android deliberately remains fail-closed until either local mobile inference or a cryptographically secure paired runtime is available. A remote target must use HTTPS except for loopback and its pairing token is memory-only.

v0.50 adds the local-inference foundation underneath that shell: a Python-free mobile package (`contract.json + weights.safetensors`) and a pure-Rust one-token factorized/recurrent kernel. CI compares Rust logits/state against a deterministic Python golden trajectory and also compiles the kernel for `aarch64-linux-android`. Tokenization, autoregressive sampling and Tauri local-runtime wiring are still open, so the APK is not yet claimed production-complete.

v0.51 adds a separate native generation host on top of that kernel. It loads the frozen Hugging Face `tokenizer.json` directly in Rust, performs prompt prefill and bounded greedy autoregressive generation, and is checked against Python tokenization/generation goldens. Exact product chat-template rendering, sampling, persistent state bridging and Tauri local routing remain open.

v0.52 freezes the exact pinned product chat-template result into an integrity-bound prefix/between/suffix contract. Rust verifies the contract and tokenizer hashes, renders system+user prompts without shipping Jinja, and is checked byte-for-byte and token-for-token against Python Transformers on the exact pinned Qwen3 tokenizer revision. Dynamic product state/profile/memory construction, sampling and Tauri LocalMobile routing remain open.

## Browser UI court

The frontend includes a deterministic browser-preview backend only for UI verification. It is never packaged as inference authority.

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/product_ui_browser.py
```

The court verifies desktop and phone viewports, power semantics, chat interaction, personalization persistence, minimum touch targets, horizontal overflow and reduced motion.

## Product runtime courts

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_product_runtime.py tests/test_product_release_assets.py
```

These cover actual LivingEngine integration, power state, transcript persistence, memory-off privacy behavior, loopback auth, release asset staging and L36 authority binding.

## Native development build

Install the current Tauri CLI, then:

```bash
cd apps/product-client
tauri build --no-bundle
```

Android:

```bash
cd apps/product-client
tauri android init
tauri android build --apk --target aarch64
```

Tauri's generated `src-tauri/gen/` tree is intentionally not committed.

## Windows release build

A production installer must not be produced from arbitrary local files.

Use the **Windows Product Release** workflow. It requires:

- promoted factorized model URL;
- exact model SHA-256;
- matching COMPLETE L36 ceremony JSON;
- tokenizer archive URL + SHA-256.

The workflow:

1. downloads assets only from HTTPS;
2. builds the self-contained PyInstaller sidecar;
3. verifies model SHA and L36 ceremony;
4. stages sidecar/model/tokenizer as Tauri resources;
5. starts the bundled runtime;
6. powers the model on and performs a real chat inference smoke;
7. builds an NSIS one-click installer with offline WebView2 installation support;
8. uploads the installer plus release manifest.

Missing or mismatched evidence stops the release.

## Release boundary

v0.42 establishes the product/distribution substrate. A Windows installer is only a real release once the manual release workflow runs with an actually promoted checkpoint and succeeds end-to-end.

Android UI/APK compilation is not the same claim as Android local inference. That remains explicitly open.

### Dynamic product payload parity

v0.53 freezes the personalization/state/memory payload below the existing UI. Desktop Python and the native mobile runtime now share one structured payload schema and are required to render the same pinned-Qwen prompt and token IDs. No additional settings panel or Android-only chat surface is introduced.

The current native parity covers preferred name, language, response length, conversational style, personal instruction, relationship/affect state, four unresolved threads, eight relevant memories and reply/initiative task wording. Sampling and persistent Android state remain separate follow-on gates.

v0.54 freezes seeded stochastic decoding parity below the same Ember Quiet UI. Desktop Python and native Rust share SplitMix64 + quantized nucleus sampling; CI freezes generated sampled token IDs, text, EOS behavior and final recurrent state. The production desktop path still chooses a fresh 64-bit seed per response unless a deterministic seed source is injected for courts/replay.

Android is still not called production-complete until persistent local state/latent binding, Tauri LocalMobile routing, authorized release assets and emulator/device courts close.

