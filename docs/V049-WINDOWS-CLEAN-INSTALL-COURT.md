# v0.49 Windows Clean-Install Court

## Goal

A successful installer build is not enough for v1.

v0.49 requires the exact packaged Windows product to survive a clean
installation and prove that the installed assets can run the local AI.

No new product UI is added.

## Authority-bound installed resources

The release workflow already stages:

- self-contained PyInstaller runtime;
- promoted factorized checkpoint;
- tokenizer assets;
- COMPLETE promotion ceremony.

v0.49 additionally embeds the generated `release-assets.json` inside the
installed model resources while preserving the external copy uploaded with the
release artifact.

The installed copy binds:

- runtime executable SHA-256;
- model SHA-256;
- promotion ceremony SHA;
- tokenizer asset hashes.

## Clean-install sequence

After NSIS is built, the Windows release workflow:

1. silently installs into an empty temporary directory;
2. finds exactly one installed:
   - `Nolane.exe`;
   - `nolane-product-runtime.exe`;
   - `factorized-nolane.pt`;
   - promotion ceremony;
   - tokenizer config + tokenizer JSON;
   - release-assets manifest;
3. checks installed runtime/model hashes against the installed manifest;
4. checks the installed ceremony authorizes the installed model;
5. checks installed application version equals the release version;
6. launches the **installed** sidecar from the installed tree;
7. requires authenticated readiness PASS;
8. powers the installed model on;
9. performs one real local chat;
10. verifies local history persisted;
11. stops the direct sidecar;
12. launches the installed `Nolane.exe`;
13. requires the app to remain alive and spawn a new bundled runtime process.

Only then does the release workflow emit a PASS receipt.

## Receipt privacy

`NOLANE-V049-WINDOWS-CLEAN-INSTALL-RECEIPT-V1` records hashes and boolean
outcomes.

It does not record:

- chat text;
- auth token;
- user data directory.

The receipt is uploaded beside the installer and release-assets manifest.

## PR-time safety

The actual clean install needs a real promoted checkpoint, so it belongs to the
manual release workflow.

Ordinary Product Client CI still parses the PowerShell court using PowerShell's
own AST parser. A syntax-broken clean-install script therefore cannot merge.

Release asset staging tests also prove the manifest embedded into
`resources/model/` exactly matches the external release manifest.

## v1 boundary

v0.49 closes installer mechanics only when the manual workflow is run with a
real L36-authorized checkpoint and returns PASS.

Source code and synthetic CI cannot substitute for that real release evidence.
