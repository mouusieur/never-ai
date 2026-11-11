"""Native Windows (.exe) & Cross-Platform Desktop Launcher for NEVER-AI."""

from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import uvicorn


class _OTASourceFirstFinder(importlib.abc.MetaPathFinder):
    """Ensure OTA-updated `never_ai.*` modules in `never_ai_data/src` override frozen PyInstaller bytecode."""

    def __init__(self, ota_src_dir: Path) -> None:
        self.ota_src_dir = ota_src_dir

    def find_spec(self, fullname: str, path: object = None, target: object = None):  # type: ignore[override]
        if fullname != "never_ai" and not fullname.startswith("never_ai."):
            return None
        if not (self.ota_src_dir / "never_ai").is_dir():
            return None
        if "." not in fullname:
            return importlib.machinery.PathFinder.find_spec(fullname, [str(self.ota_src_dir)], target)
        rel_parent = Path(*fullname.split(".")[:-1])
        search_dir = self.ota_src_dir / rel_parent
        if search_dir.is_dir():
            return importlib.machinery.PathFinder.find_spec(fullname, [str(search_dir)], target)
        return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_user_data_dir() -> Path:
    """Store weights, conversations & local profiles cleanly in portable or AppData dir."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        portable_dir = exe_dir / "never_ai_data"
        try:
            portable_dir.mkdir(parents=True, exist_ok=True)
            return portable_dir
        except OSError:
            appdata = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NeverAI"
            appdata.mkdir(parents=True, exist_ok=True)
            return appdata
    return Path.cwd()


def _enable_windows_dark_titlebar() -> None:
    """Apply Windows 10/11 pitch-black DWM immersive dark mode to active window."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd:
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            val = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(val),
                ctypes.sizeof(val),
            )
    except Exception:  # noqa: BLE001
        pass


def _wait_for_server(url: str, timeout_s: float = 25.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/status", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.15)
    return False


def _launch_chromeless_browser_app(url: str) -> subprocess.Popen[bytes] | None:
    """Fallback native chromeless --app window via Edge WebView / Chromium."""
    candidates: list[str] = []
    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates.extend(
            [
                os.path.join(pfx86, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
            ]
        )
    else:
        for b in ("google-chrome", "chromium", "microsoft-edge"):
            found = shutil.which(b)
            if found:
                candidates.append(found)

    for bin_path in candidates:
        if os.path.exists(bin_path):
            return subprocess.Popen([bin_path, f"--app={url}", "--window-size=1360,860"])
    return None


def main() -> None:
    data_dir = _resolve_user_data_dir()
    os.chdir(data_dir)

    # Register OTA source-first finder BEFORE PyInstaller's FrozenImporter so remote updates apply without reinstalling!
    local_src = data_dir / "src"
    if str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    sys.meta_path.insert(0, _OTASourceFirstFinder(local_src))

    # Automatic Zero-Reinstall Over-The-Air (OTA) Remote Update Check on Boot!
    ota_instance = None
    try:
        from never_ai.updater.ota import OTAUpdater

        ota_instance = OTAUpdater(app_root=data_dir)
        if ota_instance.state.get("auto_update_on_boot", True) and not (data_dir / ".git").exists():
            ota_instance.apply_remote_update(force=False)
    except Exception:  # noqa: BLE001
        pass

    # Pre-initialize engine & OTA controller in data_dir
    import never_ai.server.app as server_app
    from never_ai.engine import NeverAIEngine

    if ota_instance is not None:
        server_app._OTA = ota_instance

    server_app._ENGINE = NeverAIEngine(
        vault_dir=data_dir / "checkpoints",
        memory_path=data_dir / "memory_store" / "evolvemem.json",
        telemetry_path=data_dir / "logs" / "training_telemetry.jsonl",
        seq_len=128,
    )

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    cfg = uvicorn.Config(
        server_app.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    srv = uvicorn.Server(cfg)
    srv_thread = threading.Thread(target=srv.run, daemon=True)
    srv_thread.start()
    _wait_for_server(url)

    try:
        import webview  # type: ignore[import-not-found]

        window = webview.create_window(
            title="NEVER-AI",
            url=url,
            width=1360,
            height=860,
            min_size=(900, 620),
            background_color="#000000",
        )
        webview.start(_enable_windows_dark_titlebar)
        _ = window
    except ImportError:
        proc = _launch_chromeless_browser_app(url)
        if proc is not None:
            proc.wait()
        else:
            import webbrowser

            webbrowser.open(url)
            try:
                while srv_thread.is_alive():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
