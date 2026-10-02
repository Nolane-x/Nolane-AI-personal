# Release model resources

Production Windows builds stage exactly two authority-bound files here:

- `factorized-nolane.pt`
- `promotion-ceremony.json`

The model file is not accepted merely because its filename or SHA looks correct.
`scripts/stage_product_release_assets.py` verifies a **COMPLETE L36 promotion
ceremony** and requires its `candidate_checkpoint_sha256` to match the exact
model bytes.

The product sidecar repeats that verification on startup before any model can
be powered on.

Model weights and real ceremony evidence are intentionally not committed to
normal Git history.
