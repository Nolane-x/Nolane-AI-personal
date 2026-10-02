# v0.42 Product Client — NUI Contract

## Evidence class

`ARTIFACT_WORK`.

This work ships a real product client. It does not claim that Nolane UI
Intelligence causally improves UI quality.

## Task-profile checksum

- product: Nolane AI Personal
- primary job: talk to one persistent personal AI with near-zero interface friction
- secondary job: turn the AI on/off and quietly personalize how it relates and speaks
- platforms: Windows desktop and Android phone only
- modalities: keyboard/mouse + touch + software keyboard
- visual ambition: exceptional but deliberately restrained
- visual anchor: orange
- product density: very low
- privacy posture: local-first; no account/cloud surface required by default
- AI control: obvious power state; no hidden autonomous-on state
- personalization: high depth, low surface area
- Windows distribution: one installer must contain the app/runtime prerequisites;
  missing model/tokenizer release assets must fail the production build rather
  than produce a non-chatting installer
- Android distribution: same chat surface and state model; mobile local inference
  is not claimed until a supported mobile inference path exists
- user-visible non-goals: IDE, terminal, tool graph, multi-agent supervisor,
  dashboard, prompt playground, model picker, exposed evidence courts

Checksum-changing facts require rerouting.

## NUI route-justification ledger

Activated owners:

- human-AI interaction: AI state and agency are core product semantics
- AI autonomy/control: the user explicitly asked for a clear on/off control
- latency/progressive feedback: local model load and generation can take time
- desktop windowed workspace: Windows is a first-class target
- responsive/mobile interface: Android is a first-class target
- touch + keyboard input: both platforms must preserve action equivalence
- theming/personalization: deep personalization must remain visually quiet
- privacy-sensitive UI: personal memory/profile state is private
- offline/degraded experience: Windows local runtime can be unavailable/loading
- accessible interfaces: product-critical chat/composer/power controls
- typography/color/layout/motion: exceptional visual quality was requested
- functional completeness + runtime verification: this is intended to ship

Deliberately inactive:

- multi-agent surfaces: not in product contract
- code editor / IDE / terminal / browser: not in product contract
- charts/data visualization: no user job requires them
- drag/drop/canvas/spatial UI: no user job requires them
- commerce/authentication: no account or purchase flow is requested
- rich text editor: composer is plain conversational input
- heavy command palette/navigation: conflicts with the low-surface-area contract

## NUI V12.1 reference execution capsule

Task fingerprint:

`nolane-personal-v042|windows+android|ai-chat|minimal|orange|power-control|personalization|local-first`

Reference posture: `ACTIVE`.

Required packs:

- `ai-chat`
- `native-component-system`
- `button-feedback`
- `loading-success-error`
- `accessibility-verification`
- `performance-quality`

Preserved source IDs used as mechanism references only:

- `assistant-ui`
- `ai-sdk`
- `tamagui`
- `radix-primitives`
- `playwright`
- `lighthouse`

Adoption posture:

- no third-party UI component code is copied into the product
- referenced projects influence only bounded mechanisms such as transcript
  hierarchy, state feedback, focus behavior, touch sizing and verification
- local UI primitives are independently implemented
- Tauri is a platform/runtime dependency, not a visual design authority

License gate: no restrictive direct-adoption candidate selected.

Open reference verification:

- re-check final Windows installer behavior
- inspect Android safe-area/IME behavior on a real emulator/device
- inspect reduced-motion behavior
- run keyboard/focus and touch-target courts
- inspect rendered screenshots at desktop and phone sizes

## Divergence record

### Direction A — Ember Quiet — selected

Warm neutral canvas, nearly invisible chrome, orange used only for living state
and primary action. Assistant transcript is document-like. User messages are
compact right-aligned soft-orange surfaces. The composer is one quiet floating
bar.

Signature: a single living ember power control.

Restraint rule: no second animated decorative object may compete with the ember.

### Direction B — Ember Instrument

Dark charcoal instrument surface, stronger luminous orange status language,
more persistent runtime telemetry.

Rejected because it makes the runtime machinery too visible and turns a
companion into an instrument panel.

### Direction C — Solar Paper

Editorial cream paper, large conversational typography and more expressive
orange section rhythm.

Rejected because it makes short everyday chat feel more like reading a product
page than talking naturally.

## Selected interaction architecture

Always visible:

1. tiny product identity
2. AI power control/status
3. conversation
4. composer

Hidden until requested:

- personalization sheet
- mobile/connection diagnostics
- privacy/memory controls
- advanced runtime diagnostics

No permanent sidebar.

## Personalization contract

The visible chat stays stable while behavior can be personalized through:

- preferred name
- language: auto / Vietnamese / English
- response length: compact / balanced / expansive
- conversational style: natural / warm / direct / playful
- proactive initiative: off / gentle / active
- memory: enabled / disabled
- optional personal instruction

Personalization changes language-model context and living-runtime behavior, not
only visual labels.

## AI power semantics

OFF:

- model must not accept a chat request
- power control is visually unambiguous
- local state/history remains readable

STARTING:

- one visible progress state
- composer disabled
- repeated power presses cannot spawn duplicate model instances

ON:

- composer enabled
- model identity/checkpoint status available to diagnostics
- new requests may execute

THINKING:

- the living ember may use a restrained pulse
- send is disabled for duplicate submission
- transcript remains readable

ERROR:

- state is explicit
- user receives one recovery action
- no fake ON indicator

## Visual system

Primary orange: `#F26A2E`.

Orange is not used for long text, low-contrast decoration or every icon.

Surfaces:

- light: warm off-white / soft stone / charcoal text
- dark: near-black warm neutral / soft white text
- orange remains the semantic living/action accent

Typography:

- system UI stack only; no font download required
- conversational body optimized for reading rather than brand display
- maximum chat measure is bounded on desktop
- phone layout uses full useful width while respecting safe areas

Geometry:

- soft radii, not pill-everything
- touch targets >=44 CSS px
- composer and power control receive the strongest interaction affordance
- no card grid

Motion:

- 120–220 ms state transitions
- only state-bearing motion
- reduced-motion mode removes pulsing and translates transitions to opacity

## Product obligations

- first launch reaches a usable chat shell without setup wizard sprawl
- AI power state survives frontend reload coherently with backend truth
- chat history restores from local runtime
- Enter sends; Shift+Enter inserts newline on desktop
- Android IME does not hide the composer
- power control has accessible name and state
- focus order is identity -> power -> transcript controls if any -> composer -> send
- transcript uses semantic live-region behavior without rereading the whole log
- errors never appear only as color
- personalization sheet can be closed with Escape/back
- user can disable memory without losing access to existing transcript
- UI never claims Android local inference until a mobile runtime exists

## Completion gate

v0.42 may be called product-client complete only when:

- source/static contract tests pass
- Python product-runtime tests pass
- Windows sidecar build court passes
- Windows Tauri compile/bundle court passes
- Android Tauri compile court passes
- browser/render smoke checks pass at desktop and phone viewports
- two critique/correction cycles are recorded
- no unresolved critical/major NUI finding remains

If release model/tokenizer assets are missing, the **release installer** remains
BLOCKED even when client/runtime engineering is otherwise complete.
