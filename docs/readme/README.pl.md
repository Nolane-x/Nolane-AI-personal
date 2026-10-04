# Nolane AI Personal

**Osobista AI local-first zaprojektowana tak, aby zachowywać ciągłość w czasie zamiast zaczynać od nowa przy każdym prompcie.**

[English](../../README.md) · **Polski**

## v1.0

- **Nolane Presence** — mała kula jest abstrakcyjną twarzą i delikatnie odzwierciedla rzeczywisty stan.
- **Pamięć i otwarte wątki** — zobacz, co Nolane pamięta i do czego może wrócić; wspomnienia można zachować, edytować lub zapomnieć.
- **Rozwój relacji** — „Nowa znajomość → Znajomi → Blisko”, bez XP i poziomów.
- **Trzy kroki pierwszego uruchomienia** — preferowane imię, język interfejsu i styl rozmowy.
- **Proaktywna rozmowa bez przeszkadzania** — kiedy Nolane chce coś powiedzieć, pojawia się mała kapsuła; zignorowana nie przerywa pracy.
- **Subtelna atmosfera** — tło zmienia się bardzo nieznacznie wraz z mood.
- Zmiana nazwy i awatara nie zmienia checkpointu ani runtime identity.

## Prywatność local-first

Stan, pamięć i ciągłość są lokalne. Observable Mind pokazuje tylko ograniczone podsumowania stanu, nigdy surowe ukryte rozumowanie. Dane do uczenia wymagają jawnej akceptacji użytkownika.

## Platformy

Windows: Tauri + lokalny runtime. Android: Tauri + natywny Rust LocalMobile. Linux: narzędzia runtime/research. macOS/iOS nie są celami v1.

## Rozwój

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

Zobacz [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) i [v1 Living Presence](../V1-LIVING-PRESENCE.md).

Wydanie v1 wymaga CI-verified closure dla tego samego commit i nie deklaruje uniwersalnej certyfikacji baterii, termiki ani OEM.

Apache-2.0.
