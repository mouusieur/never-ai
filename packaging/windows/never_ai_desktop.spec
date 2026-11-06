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
        "never_ai.updater.ota",
        "never_ai.plugins.web_search",
        "never_ai.data.conversational_universe",
        "sympy",
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
    upx=False,  # CRITICAL: Disable UPX compression to prevent Windows Defender / Chrome heuristic false positives!
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "assets" / "brand" / "never_ai.ico"),
    version=str(ROOT / "packaging" / "windows" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NeverAI",
)
