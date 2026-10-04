# Nolane AI Personal

**Un'IA personale local-first progettata per mantenere continuità nel tempo, invece di ripartire come un nuovo chatbot a ogni prompt.**

[English](../../README.md) · **Italiano**

## v1.0

- **Nolane Presence** — un piccolo globo funge da volto astratto e riflette lo stato reale con luce e movimento minimi.
- **Memoria e fili aperti** — mostra ciò che Nolane ricorda e gli argomenti rimasti in sospeso; una memoria può essere conservata, modificata o dimenticata.
- **Crescita della relazione** — “Appena conosciuti → Familiari → Vicini”, senza XP o livelli.
- **Onboarding in tre passi** — nome preferito, lingua dell'interfaccia e stile di conversazione.
- **Conversazione proattiva discreta** — quando Nolane vuole parlare compare una piccola capsula; se la ignori, resta silenzioso.
- **Atmosfera sottile** — lo sfondo varia appena in base al mood.
- Nome e avatar possono cambiare senza modificare checkpoint o identità interna del runtime.

## Privacy local-first

Stato, memoria e continuità sono locali. Observable Mind espone solo riepiloghi limitati dello stato, mai ragionamento privato grezzo. I dati di apprendimento richiedono una revisione esplicita dell'utente.

## Piattaforme

Windows: Tauri + runtime locale. Android: Tauri + LocalMobile nativo Rust. Linux: strumenti runtime/ricerca. macOS/iOS non sono target v1.

## Sviluppo

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

Vedi [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) e [v1 Living Presence](../V1-LIVING-PRESENCE.md).

Il rilascio software v1 richiede la closure CI verificata sullo stesso commit e non dichiara una certificazione universale di batteria, termiche o OEM.

Apache-2.0.
