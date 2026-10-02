# Bundled runtime

Production Windows builds replace this placeholder with the PyInstaller onedir
output for `nolane-product-runtime.exe`.

A release workflow MUST fail if the executable is absent after staging.
Do not ship this placeholder as a working runtime.
