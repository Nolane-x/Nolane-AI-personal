# Nolane Product Client

Shared **Windows + Android** product surface for Nolane AI Personal v1.0.

The UI is intentionally small: identity, language, observable presence, AI power, transcript and composer. Deeper controls live in sheets instead of permanent dashboard chrome.

## v1 surface

The client implements the **Ember Quiet** NUI direction:

- **Nolane Presence** — runtime-driven orb mood/activity, with reduced-motion support;
- **Observable Mind** — safe state summary, not raw private reasoning;
- **Memory & Threads** — real local memory projection with keep/edit/forget controls;
- **Relationship growth** — New → Familiar → Close, derived from bounded relationship state;
- **three-step onboarding** — preferred name, UI language and conversation style;
- **proactive capsule** — initiative speech waits behind a quiet surface rather than a popup;
- **identity skin** — local AI name/avatar customization;
- **18 UI languages** — English, Vietnamese, Chinese, Japanese, Korean, Spanish, French, German, Portuguese, Italian, Thai, Indonesian, Russian, Arabic, Hindi, Turkish, Polish and Dutch.

The browser-preview backend exists only for deterministic UI courts and is never packaged as model authority.

## Runtime architecture

```text
Windows
Tauri WebView
   │ invoke()
   ▼
Rust native host
   │ authenticated random loopback
   ▼
bundled product runtime
   ▼
LivingEngine + Personal Cortex
   ├── local SQLite identity / memory / history
   ├── promoted factorized model
   ├── tokenizer assets
   └── release authority evidence

Android
Tauri WebView
   │ invoke()
   ▼
Rust native host
   ▼
LocalMobile
   ├── native tokenizer + prompt contract
   ├── recurrent/factorized inference
   ├── persistent identity / relationship / memory
   ├── initiative + REST baseline
   └── authority-bound mobile bundle
```

The web layer never receives the Windows loopback authentication token.

## Browser court

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/product_ui_browser.py
```

The court covers desktop/mobile viewports, first-run onboarding, language persistence, AI power truth, Observable Mind, relationship state, keep/edit/forget memory controls, chat, avatar/name persistence, reduced motion, touch targets and overflow.

## Runtime courts

```bash
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/test_product_runtime.py \
  tests/test_product_release_assets.py \
  tests/test_product_version.py \
  tests/test_v1_closure.py
```

Native Android parity is additionally courted by the Rust LocalMobile fixtures, including restart continuity and v1 memory-control persistence.

## Development

Windows:

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

Android:

```bash
cd apps/product-client
tauri android init
tauri android build --apk --target aarch64
```

Generated `src-tauri/gen/` trees are intentionally not committed.

## Release integrity

Product version must match across:

- root `pyproject.toml`;
- `apps/product-client/package.json`;
- Tauri `Cargo.toml`;
- `tauri.conf.json`.

The v1.0 software release gate is the **same-SHA CI closure** across Product Client, Living Runtime, Neural Shadow and Platform Crash. Product Client evidence includes Windows native packaging, Android arm64/x86_64 packaging, native LocalMobile execution and packaged emulator force-stop/restart continuity.

Physical-device battery, thermal behavior, OEM compatibility and real-world performance distributions remain separate certification claims. The software release does not imply those claims.

See [the repository README](../../README.md) and [v1 Living Presence](../../docs/V1-LIVING-PRESENCE.md).
