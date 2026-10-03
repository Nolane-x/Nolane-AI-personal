# v0.42 NUI Critique Ledger

Evidence class: `ARTIFACT_WORK`.

This ledger records critique/correction work separately from generation. It
does not certify release by itself. The Product Client Court supplies rendered
evidence for the final gate.

## Cycle 1 — interaction geometry

### Finding

The first Ember Quiet implementation used 42px send and close controls while
the v0.42 NUI product contract required >=44 CSS px primary touch targets.

This was a real contract defect even though the controls looked visually
balanced on desktop.

### Correction

- composer action column: 42 -> 44px
- send button: 42 -> 44px
- dialog close button: 42 -> 44px
- primary/secondary action minimum height: 42 -> 44px

### Re-observation obligation

`tests/product_ui_browser.py` measures the rendered power and send controls
at desktop and mobile viewports and fails below 44px.

Status: correction implemented; final rendered PASS is owned by CI.

## Cycle 2 — product truth, privacy and release authority

### Finding A: memory-off semantics

The UI memory toggle originally stopped new episodic writes/retrieval but the
background REST subsystem could still process older memories.

That violates the user-facing meaning of disabling memory.

### Correction A

ProductRuntime now disables both:

- LivingEngine memory retrieval/write;
- REST/consolidation scheduling.

Transcript history remains readable because conversation events and memory
policy are deliberately separate layers.

### Finding B: local runtime authority

The first Windows host relied on loopback binding alone.

For a personal AI containing private transcript/memory state, another local
process should not gain API access merely because it can reach 127.0.0.1.

### Correction B

Every Windows launch now creates a 256-bit OS-random authentication token.
Only the Rust host receives it and injects it into proxied runtime requests.
The WebView never receives the token.

### Finding C: release candidate truth

Initial release staging verified the model SHA but did not prove that the
checkpoint had production promotion authority.

A byte-identical unpromoted candidate could therefore have been bundled if an
operator supplied its SHA.

### Correction C

Release staging and runtime startup now require a self-digested COMPLETE L36
promotion ceremony whose candidate checkpoint SHA equals the bundled model
bytes. The ceremony is bundled beside the model and reverified at startup.

### Re-observation obligation

Product runtime/release courts verify:

- memory off disables actual memory + REST behavior;
- token-protected HTTP access;
- wrong ceremony/checkpoint binding is rejected;
- the matching ceremony is carried into release resources.

Status: corrections implemented; CI PASS still required.

## Cycle 3 — mobile interaction layering

### Finding

The touch-mode browser court showed that the mobile personalization sheet's
scrolling body could sit above the header in the hit-test stack. The close
button was visually present and geometrically large enough, but the body
intercepted taps.

This is a real mobile interaction defect because a visible close control that
cannot receive touch does not satisfy the product contract.

### Correction

- sheet header/footer now establish an explicit higher stacking layer;
- the scrolling sheet body remains below that layer;
- the close button receives its own top interaction layer;
- no force-click or test-only bypass is used.

### Verification

The mobile Playwright court must close the sheet with a real touch `tap()` on
`#closeProfileButton`.

Status: correction implemented; final rendered PASS is owned by CI.

## Render evidence

The NUI browser job captures:

- `desktop-chat.png`
- `desktop-personalization.png`
- `mobile-chat.png`
- `mobile-personalization.png`

These are uploaded as the `product-ui-render-evidence` workflow artifact.

A green DOM/source test without these rendered observations is not enough for
the bounded v0.42 UI claim.

## Open findings that remain intentionally non-waived

- Android local inference is not implemented.
- Android production pairing lacks a finished TLS/pinning/key-storage ceremony.
- A real Windows one-click release requires an actually promoted checkpoint,
  real tokenizer assets and successful Windows Product Release workflow.
- Clean-VM installer smoke remains required before v1.0 distribution closure.

These are roadmap boundaries, not hidden PASS states.
