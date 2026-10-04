# Nolane AI Personal

**Une IA personnelle local-first conçue pour conserver une continuité dans le temps, plutôt que de repartir de zéro à chaque prompt.**

[English](../../README.md) · **Français**

## v1.0

- **Nolane Presence** — un petit orbe sert de visage abstrait et reflète l’état réel avec des variations très légères.
- **Mémoire et sujets en cours** — consultez ce que Nolane retient et les sujets non terminés ; une mémoire peut être conservée, modifiée ou oubliée.
- **Évolution de la relation** — « Nouvelle rencontre → Familier → Proche », sans XP ni niveaux.
- **Onboarding en trois étapes** — nom préféré, langue de l’interface et style de conversation.
- **Conversation proactive discrète** — une petite capsule apparaît lorsque Nolane souhaite parler ; l’ignorer suffit pour rester au calme.
- **Atmosphère subtile** — l’arrière-plan varie à peine selon le mood.
- Le changement de nom ou d’avatar ne modifie ni le checkpoint ni l’identité interne du runtime.

## Confidentialité local-first

L’état, la mémoire et la continuité sont locaux. Observable Mind expose uniquement des résumés d’état bornés, jamais le raisonnement privé brut. Les données d’apprentissage nécessitent une revue explicite de l’utilisateur.

## Plateformes

Windows : Tauri + runtime local. Android : Tauri + LocalMobile natif en Rust. Linux : outils runtime/recherche. macOS/iOS ne sont pas des cibles v1.

## Développement

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

Voir [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) et [v1 Living Presence](../V1-LIVING-PRESENCE.md).

La sortie logicielle v1 exige une closure CI vérifiée sur le même commit et ne prétend pas certifier universellement batterie, thermique ou OEM.

Apache-2.0.
