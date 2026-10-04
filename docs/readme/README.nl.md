# Nolane AI Personal

**Een local-first persoonlijke AI die continuïteit door de tijd bewaart, in plaats van bij elke prompt als een nieuwe chatbot te beginnen.**

[English](../../README.md) · **Nederlands**

## v1.0

- **Nolane Presence** — een kleine orb werkt als abstract gezicht en weerspiegelt echte toestand met zeer subtiele licht- en bewegingsveranderingen.
- **Geheugen en open onderwerpen** — bekijk wat Nolane onthoudt en welke gesprekken nog openstaan; herinneringen kunnen worden bewaard, bewerkt of vergeten.
- **Relatiegroei** — “Net ontmoet → Vertrouwd → Hecht”, zonder XP of levels.
- **Onboarding in drie stappen** — voorkeursnaam, interfacetaal en gesprekstijl.
- **Proactief zonder te storen** — als Nolane iets wil zeggen verschijnt alleen een kleine capsule; negeren houdt de ervaring stil.
- **Subtiele atmosfeer** — de achtergrond verandert slechts licht met de mood.
- Naam en avatar kunnen wijzigen zonder checkpoint of runtime identity te veranderen.

## Local-first privacy

Toestand, geheugen en continuïteit zijn lokaal gericht. Observable Mind toont alleen begrensde toestandsamenvattingen en nooit ruwe verborgen redenering. Leerdata vereist expliciete beoordeling door de gebruiker.

## Platforms

Windows: Tauri + lokale runtime. Android: Tauri + native Rust LocalMobile. Linux: runtime/research tooling. macOS/iOS zijn geen v1-doelen.

## Ontwikkeling

```bash
python -m pip install -e .
nolane-personal init
nolane-personal run --no-model --tick-seconds 5
```

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

Zie [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) en [v1 Living Presence](../V1-LIVING-PRESENCE.md).

Een v1-softwarerelease vereist CI-verified closure op dezelfde commit en claimt geen universele batterij-, thermische of OEM-certificering.

Apache-2.0.
