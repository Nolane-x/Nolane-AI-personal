# LocalMobile resource namespace

v0.56 wires the Android Tauri shell to a native Rust LocalMobile product host.

Production release tooling will later stage the following L36-bound assets here:

- `localmobile-manifest.json`
- `bootstrap-state.json`
- `package/`
- `tokenizer.json`
- `tokenizer_config.json`
- `prompt-contract.json`

Source builds intentionally do not commit model weights or promotion authority.
If these release assets are absent, Android exposes a clear LocalMobile runtime
error and may still be manually pointed at a remote runtime for development.
