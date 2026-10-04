# Nolane AI Personal

<p align="center">
  <strong>A local-first personal AI designed to live through time, not reset after every prompt.</strong>
</p>

<p align="center">
  Windows · Android · Local memory · Proactive conversation · Persistent identity · Apache-2.0
</p>

> **v1.0.0** turns the research runtime into a deliberately small personal product. Nolane keeps a persistent identity, relationship state, memories, unfinished threads and initiative while its language cortex remains a bounded component rather than the owner of the person-like state.

## Languages

**English** · [Tiếng Việt](docs/readme/README.vi.md) · [中文](docs/readme/README.zh.md) · [日本語](docs/readme/README.ja.md) · [한국어](docs/readme/README.ko.md) · [Español](docs/readme/README.es.md) · [Français](docs/readme/README.fr.md) · [Deutsch](docs/readme/README.de.md) · [Português](docs/readme/README.pt.md) · [Italiano](docs/readme/README.it.md) · [ไทย](docs/readme/README.th.md) · [Bahasa Indonesia](docs/readme/README.id.md) · [Русский](docs/readme/README.ru.md) · [العربية](docs/readme/README.ar.md) · [हिन्दी](docs/readme/README.hi.md) · [Türkçe](docs/readme/README.tr.md) · [Polski](docs/readme/README.pl.md) · [Nederlands](docs/readme/README.nl.md)

The product UI ships with the same 18 interface languages.

## Why Nolane is different

Most assistants reconstruct a temporary persona from the current prompt and chat history. Nolane separates **language generation** from **continuity**.

```text
events + time
     │
     ▼
Living Runtime
identity · affect · relationship · threads · memory · initiative
     │
     ├── Observable Mind ──► safe state summary, never raw private reasoning
     ├── REST             ──► bounded evidence-backed consolidation
     └── Personal Cortex  ──► language generation
```

That separation gives the product several useful properties:

- **Persistent identity** — the same local identity and state survive restart.
- **Relationship continuity** — familiarity, trust and closeness evolve from interactions rather than a fake level counter.
- **Local memory** — memory participates in generation and can be kept, edited or forgotten by the user.
- **Open threads** — unfinished topics can remain available for later follow-up.
- **Real initiative** — Nolane can choose to speak or remain silent without a fresh user prompt.
- **Bounded observability** — the UI can show mood, intent, uncertainty and active context without exposing hidden chain-of-thought.

## v1.0 product experience

### Nolane Presence

The small Nolane orb acts as an abstract face rather than a human avatar or emoji. Its light and motion are driven by actual runtime state:

- warm/playful states become subtly brighter;
- concern/low-energy states soften;
- curiosity gets restrained motion;
- thinking pulses;
- resting becomes almost still.

The surrounding conversation atmosphere changes only a few percent with mood. It is intentionally subconscious rather than a theme switch.

### Memory & Threads

Observable Mind shows two compact views:

- **unfinished threads** — subjects Nolane may naturally return to;
- **what Nolane remembers** — local memories currently available to the cortex.

A memory can be **kept**, **edited**, or **forgotten**. These actions change the real local retrieval state on desktop and Android LocalMobile. Desktop memory controls also leave digest-only audit records; raw deleted memory text is not copied into that control audit.

### Relationship growth

Nolane does not show XP, streaks, hearts or levels. The UI derives one quiet human-readable stage from the runtime relationship state:

```text
New → Familiar → Close
```

The underlying closeness, trust and familiarity remain bounded runtime signals.

### Three-step first run

The first launch asks only:

1. what Nolane should call you;
2. the interface language;
3. how Nolane should speak.

The result is stored locally in the existing product profile and can be changed later.

### Proactive conversation, without pop-ups

When the initiative engine produces a message while you are not actively chatting, the client does not interrupt with a modal. It presents a small capsule such as **“Nolane has something to say.”**

Open it to reveal the generated message. Ignore or dismiss it and the conversation remains quiet.

### Identity skin

You can rename Nolane and choose an avatar without changing the underlying model checkpoint, neural memory or runtime identity. Custom avatars are center-cropped, resized to **256×256**, encoded as WebP and stored locally.

## Platforms

| Platform | v1 path |
| --- | --- |
| Windows | Tauri desktop client + bundled local product runtime |
| Android | Tauri client + native Rust LocalMobile inference path |
| Linux | Runtime/research tooling; not a v1 packaged product target |
| macOS / iOS | Not a v1 target |

The Windows and Android clients share the same compact NUI product surface while keeping platform-specific runtime paths.

## Local-first privacy model

Nolane is designed so continuity does not depend on a cloud profile.

- conversation state and memory are local;
- identity survives restart locally;
- memory can be disabled at the runtime policy boundary;
- learning data is never approved merely because a conversation happened;
- reviewed learning remains an explicit local human-consent workflow;
- Observable Mind exposes state summaries, not raw hidden reasoning;
- release receipts contain hashes/status rather than private conversation text.

Remote pairing remains an explicit development/fallback path and is not silently enabled.

## Language cortex and model assets

The Windows production language cortex is **Qwen3.5-2B**. Qwen3-0.6B has been retired from every active product/runtime path after failing the product-quality bar. Its lock remains only under `config/legacy-qwen3-0.6b-model.lock.json` so historical research courts stay reproducible.

The production Windows release pins the upstream Qwen checkpoint and an exact Q8_0 GGUF conversion:

```text
upstream: Qwen/Qwen3.5-2B
revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
GGUF: bartowski/Qwen_Qwen3.5-2B-GGUF
GGUF revision: 7d26695454df6de5fbcce2e58681e62dae06ce43
file: Qwen_Qwen3.5-2B-Q8_0.gguf
license: Apache-2.0
```

The GGUF is hash-pinned and runs through the pinned local llama.cpp runtime. The repository's root `model.lock.json` remains a Qwen3-1.7B research-compatibility scaffold for the existing Transformer surgery/mobile tokenizer research path; it is not the Windows production model.

Model weights are intentionally **not committed to GitHub**. Historical Qwen3-0.6B evidence stays reproducible but can no longer be selected accidentally by the normal product/runtime path.

## Developer quick start

### Runtime only

```bash
python -m pip install -e .
nolane-personal init
nolane-personal status
```

Run the state, memory and heartbeat system without loading a language model:

```bash
nolane-personal run --no-model --tick-seconds 5
```

Run with the optional language dependencies:

```bash
python -m pip install -e '.[qwen]'
nolane-personal run
```

### Product client

The shared client lives in `apps/product-client/`.

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

Platform release builds are intentionally stricter than development: release assets, model/checkpoint identity and authority evidence are verified instead of silently substituted.

## Repository map

```text
apps/product-client/             Windows + Android product surface
src/nolane_personal/             Living Runtime and desktop product runtime
crates/nolane-mobile-runtime/    native Android LocalMobile runtime
docs/                            architecture, courts and research decisions
scripts/                         reproducible build/evidence tooling
tests/                           runtime, privacy, product and release courts
```

For the complete research lineage, architecture stages and evidence gates, see:

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [v1 Living Presence](docs/V1-LIVING-PRESENCE.md)
- [v1 Closure Gate](docs/V060-V1-CLOSURE-GATE.md)

## Release integrity

A software release is considered v1-ready only when the same commit satisfies the repository's **CI-verified closure**. The closure binds Product Client, Living Runtime, Neural Shadow and Platform Crash evidence to one SHA.

The software closure intentionally does **not** claim universal physical-device battery life, thermal behavior, OEM compatibility or real-world performance distribution. Those remain separate certification concerns.

This distinction is deliberate: the repository should say exactly what its evidence proves, and no more.

## Design principles

1. **Small surface, deep behavior.** Add behavior only when it earns its place in the product.
2. **State is real.** UI signals should project runtime truth, not decorative fake intelligence.
3. **Silence is an action.** A personal AI should be capable of deciding not to interrupt.
4. **Memory belongs to the user.** Remembering must be inspectable and controllable.
5. **No fake sentience claims.** “Living” describes persistent event-driven behavior and continuity, not biological consciousness.
6. **Fail closed for release authority.** Missing or mixed evidence must block a production claim.

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Nolane AI Personal is an experimental personal-AI system and research platform. v1.0 focuses on making continuity, memory, relationship and initiative feel coherent while keeping the interface deliberately quiet.
