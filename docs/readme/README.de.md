# Nolane AI Personal

**Eine local-first persönliche KI, die über Zeit Kontinuität bewahrt, statt bei jedem Prompt als neuer Chatbot zu beginnen.**

[English](../../README.md) · **Deutsch**

## v1.0

- **Nolane Presence** — ein kleiner Orb dient als abstraktes Gesicht und spiegelt echten Zustand mit sehr dezenter Helligkeit und Bewegung.
- **Erinnerungen & offene Themen** — du siehst, was Nolane sich merkt und welche Gespräche offen sind; Erinnerungen lassen sich behalten, bearbeiten oder vergessen.
- **Beziehungsentwicklung** — „Neu → Vertraut → Nah“, ohne XP, Level oder Streaks.
- **Drei Schritte beim ersten Start** — bevorzugter Name, UI-Sprache und Gesprächsstil.
- **Proaktive Gespräche ohne Unterbrechung** — wenn Nolane etwas sagen möchte, erscheint nur eine kleine Kapsel. Ignorierst du sie, bleibt Nolane still.
- **Subtile Atmosphäre** — der Hintergrund reagiert nur sehr leicht auf die Stimmung.
- Name und Avatar können geändert werden, ohne Checkpoint oder Runtime-Identität zu verändern.

## Local-first Datenschutz

Zustand, Erinnerung und Beziehungskontinuität sind lokal ausgerichtet. Observable Mind zeigt begrenzte Zustandszusammenfassungen und niemals rohe private Gedankengänge. Lerndaten benötigen eine ausdrückliche Nutzerfreigabe.

## Plattformen

Windows: Tauri + lokaler Runtime-Pfad. Android: Tauri + natives Rust LocalMobile. Linux: Runtime-/Research-Werkzeuge. macOS/iOS sind keine v1-Ziele.

## Entwicklung

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

Mehr unter [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) und [v1 Living Presence](../V1-LIVING-PRESENCE.md).

Ein v1-Software-Release benötigt die CI-verifizierte Closure für denselben Commit und behauptet keine universelle Akku-, Temperatur- oder OEM-Zertifizierung.

Apache-2.0.
