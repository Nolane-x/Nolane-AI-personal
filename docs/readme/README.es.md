# Nolane AI Personal

**Una IA personal local-first diseñada para mantener continuidad a través del tiempo, no para reiniciarse como un chatbot nuevo en cada prompt.**

[English](../../README.md) · **Español**

## v1.0

- **Nolane Presence** — un pequeño orbe funciona como rostro abstracto y refleja estado real con luz y movimiento mínimos.
- **Memoria y temas pendientes** — puedes ver lo que Nolane recuerda y lo que quedó abierto; una memoria se puede conservar, editar u olvidar.
- **Crecimiento de la relación** — “Recién conocidos → Familiar → Cercanos”, sin XP ni niveles.
- **Onboarding de tres pasos** — nombre, idioma de interfaz y estilo de conversación.
- **Conversación proactiva sin interrupciones** — Nolane muestra una pequeña cápsula cuando quiere hablar; si la ignoras, permanece en silencio.
- **Atmósfera sutil** — el fondo cambia apenas con el mood.
- Cambiar nombre/avatar no modifica el checkpoint ni la identidad interna del runtime.

## Privacidad local-first

Estado, memoria y continuidad viven localmente. Observable Mind muestra resúmenes acotados de estado, nunca razonamiento privado bruto. Los datos de aprendizaje requieren revisión explícita del usuario.

## Plataformas

Windows: Tauri + runtime local. Android: Tauri + LocalMobile nativo en Rust. Linux: herramientas de runtime/investigación. macOS/iOS no son objetivos de v1.

## Desarrollo

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

Consulta [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) y [v1 Living Presence](../V1-LIVING-PRESENCE.md).

La publicación v1 exige el cierre CI verificado para el mismo commit; no se presenta como certificación universal de batería, temperatura u OEM.

Apache-2.0.
