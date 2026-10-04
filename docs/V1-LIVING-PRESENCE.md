# Nolane AI Personal v1.0 — Living Presence

## Goal

v1.0 closes the product frontier around a single question:

> Can Nolane feel continuously present without turning the interface into a dashboard or pretending to expose private reasoning?

The answer is implemented as six small surfaces backed by existing runtime truth.

## 1. Presence

The Observable Mind orb is the abstract face of Nolane. Mood and activity are projected from bounded runtime state, not invented by the client.

- warm/playful: slightly brighter and more saturated;
- curious: restrained drift;
- concerned/low: softer and quieter;
- thinking: pulse;
- resting/off: still.

The conversation background receives a very small mood-dependent atmosphere. Reduced-motion preferences disable presence animation.

## 2. Memory & Threads

The Mind Panel now contains both unresolved conversation threads and a bounded projection of active local memories.

Desktop memory controls mutate the real SQLite memory store:

- **keep** raises salience and records user intent;
- **edit** changes the active memory text;
- **forget** removes the active memory and its graph links.

Control auditing stores action + timestamps + before/after digests. It does not duplicate raw deleted memory text.

Android LocalMobile exposes the same product action route against its frozen bounded persistent-memory projection.

## 3. Relationship growth

The UI derives a deliberately coarse stage from closeness, trust and familiarity:

```text
New → Familiar → Close
```

This is not a reward system and has no XP, streak or numeric level.

## 4. First-run onboarding

First launch asks only for:

1. preferred name;
2. UI language;
3. conversation style.

The product reuses the existing profile authority. Onboarding does not create a second personality store.

## 5. Proactive conversation

Initiative already exists below the UI. v1 changes presentation semantics.

Desktop initiative messages are identified by their non-reply intent. LocalMobile initiative messages are identified by their native initiative event ID. Unopened initiative speech is withheld from the visible transcript and represented by a small capsule.

- open → reveal the real generated message;
- dismiss → keep it out of the visible conversation;
- ignore → no modal, no interruption.

The capsule state is UI-local; it does not rewrite Living Runtime history.

## 6. Multilingual product surface

The frontier copy is localized across the same 18 UI locales already supported by the client:

English, Vietnamese, Chinese, Japanese, Korean, Spanish, French, German, Portuguese, Italian, Thai, Indonesian, Russian, Arabic, Hindi, Turkish, Polish and Dutch.

## Release boundary

This wave does not change the meaning of the v1 software closure. v1 readiness still requires same-SHA CI evidence from the repository's release gate. Physical-device battery, thermal and OEM certification remain separate claims.

## Non-goals

- no human-face avatar requirement;
- no fake chain-of-thought display;
- no relationship gamification;
- no popup-based proactive interruption;
- no cloud memory requirement;
- no claim of biological consciousness.
