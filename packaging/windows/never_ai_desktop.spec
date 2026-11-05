# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parents[1]

datas = [
    (str(ROOT / "src" / "never_ai" / "server" / "static"), "never_ai/server/static"),
    (str(ROOT / "assets" / "brand"), "assets/brand"),
    (str(ROOT / "DIRECTIVES.md"), "."),
]

a = Analysis(
    [str(ROOT / "src" / "never_ai" / "desktop" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.lifespan.on",
        "never_ai.server.app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeverAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Pure frameless/dark GUI app without black console window!
    disable_windowed_traceback=False,
    icon=str(ROOT / "assets" / "brand" / "never_ai.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NeverAI",
)
