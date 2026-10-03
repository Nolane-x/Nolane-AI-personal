# LocalMobile release resource namespace

v0.57 reserves this directory for an authority-bound Android LocalMobile
release bundle.

Normal source history intentionally contains no model weights or real promotion
evidence. The Android Product Release workflow stages:

- `localmobile-manifest.json`
- `promotion-ceremony.json`
- `bootstrap-state.json`
- `package/`
- `tokenizer.json`
- `tokenizer_config.json`
- `prompt-contract.json`

`scripts/stage_mobile_release_assets.py` accepts these resources only after
verifying a COMPLETE L36 promotion ceremony and binding every asset to the exact
authorized checkpoint.

The Android Tauri shell uses the strict LocalMobile loader. Source/court-only
bundles are not accepted as production authority.
