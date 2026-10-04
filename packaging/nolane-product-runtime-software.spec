# -*- mode: python ; coding: utf-8 -*-

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ENTRY = os.path.join(SPECPATH, "product_runtime_entry.py")
SRC = os.path.join(ROOT, "src")

analysis = Analysis(
    [ENTRY],
    pathex=[SRC],
    binaries=[],
    datas=[],
    hiddenimports=[],
    excludes=[
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="nolane-product-runtime",
    console=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="nolane-product-runtime",
)
