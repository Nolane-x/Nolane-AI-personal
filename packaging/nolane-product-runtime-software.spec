# -*- mode: python ; coding: utf-8 -*-

analysis = Analysis(
    ["packaging/product_runtime_entry.py"],
    pathex=["src"],
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
