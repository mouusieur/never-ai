from __future__ import annotations

import io
from pathlib import Path
import struct
import zipfile

from fastapi.testclient import TestClient

from never_ai.desktop.pe_exe_builder import SFX_ZIP_OFFSET, build_windows_setup_exe
from never_ai.engine import NeverAIEngine
import never_ai.server.app as server_app
from never_ai.updater.ota import OTAUpdater


def test_pe64_setup_exe_binary_and_embedded_payload() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe_bytes = build_windows_setup_exe(repo_root)

    # 1. Verify DOS & PE32+ x64 GUI headers
    assert exe_bytes[:2] == b"MZ"
    e_lfanew = struct.unpack_from("<I", exe_bytes, 0x3C)[0]
    assert e_lfanew == 0x80
    assert exe_bytes[e_lfanew : e_lfanew + 4] == b"PE\x00\x00"

    machine, num_sections = struct.unpack_from("<HH", exe_bytes, e_lfanew + 4)
    characteristics = struct.unpack_from("<H", exe_bytes, e_lfanew + 22)[0]
    opt_magic = struct.unpack_from("<H", exe_bytes, e_lfanew + 24)[0]
    subsystem = struct.unpack_from("<H", exe_bytes, e_lfanew + 24 + 68)[0]

    assert machine == 0x8664  # AMD64 / x86-64
    assert num_sections == 2
    assert characteristics == 0x0023
    assert opt_magic == 0x020B  # PE32+
    assert subsystem == 2  # IMAGE_SUBSYSTEM_WINDOWS_GUI

    # 2. Verify embedded ZIP payload at exact SFX_ZIP_OFFSET = 2560
    zip_slice = exe_bytes[SFX_ZIP_OFFSET:]
    assert zip_slice[:4] == b"PK\x03\x04"
    with zipfile.ZipFile(io.BytesIO(zip_slice), "r") as zf:
        names = set(zf.namelist())
        assert "setup_wizard.ps1" in names
        assert "Installer-NeverAI.bat" in names
        assert "never-ai-src/ota_manifest.json" in names
        assert "never-ai-src/src/never_ai/updater/ota.py" in names


def test_ota_updater_atomic_sync_preserves_user_data_and_checkpoints(tmp_path: Path) -> None:
    install_dir = tmp_path / "NeverAI_Installed"
    (install_dir / "src" / "never_ai").mkdir(parents=True)
    (install_dir / "src" / "never_ai" / "__init__.py").write_text("__version__ = '1.3.0'\n", encoding="utf-8")

    # Precious sovereign user state that MUST survive remote OTA updates
    ckpt_file = install_dir / "checkpoints" / "latest" / "model_wsm.safetensors"
    ckpt_file.parent.mkdir(parents=True)
    ckpt_file.write_bytes(b"PRECIOUS_NEURAL_WEIGHTS_DO_NOT_TOUCH")

    mem_file = install_dir / "memory_store" / "threads.json"
    mem_file.parent.mkdir(parents=True)
    mem_file.write_text('{"threads": ["my_private_conversations"]}', encoding="utf-8")

    venv_marker = install_dir / ".venv" / "pyvenv.cfg"
    venv_marker.parent.mkdir(parents=True)
    venv_marker.write_text("home = C:\\Python311\n", encoding="utf-8")

    # Build a simulated GitHub codeload branch ZIP with updated code + decoy user dirs
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("never-ai-main/src/never_ai/__init__.py", "__version__ = '1.5.0'\n")
        zf.writestr("never-ai-main/src/never_ai/engine.py", "# engine v1.5\n")
        zf.writestr("never-ai-main/src/never_ai/remote_feature.py", "REMOTE_OK = True\n")
        zf.writestr("never-ai-main/pyproject.toml", "[project]\nname='never-ai'\nversion='1.5.0'\n")
        # Decoys inside remote archive must NEVER overwrite local user checkpoints or memories!
        zf.writestr("never-ai-main/checkpoints/latest/model_wsm.safetensors", b"CORRUPTED_REMOTE_DECOY")
        zf.writestr("never-ai-main/memory_store/threads.json", '{"threads": []}')

    ota = OTAUpdater(app_root=install_dir)
    new_sha = "abcdef1234567890abcdef1234567890abcdef12"
    result = ota.apply_zip_bytes(buf.getvalue(), new_commit_sha=new_sha, branch="main")

    assert result["ok"] is True
    assert result["updated"] is True
    assert result["new_commit"] == new_sha
    assert ota.status()["installed_commit"] == new_sha

    # Code WAS updated remotely without reinstalling
    assert "__version__ = '1.5.0'" in (install_dir / "src" / "never_ai" / "__init__.py").read_text(encoding="utf-8")
    assert (install_dir / "src" / "never_ai" / "remote_feature.py").exists()

    # Local brain weights, conversations, and .venv are 100% preserved!
    assert ckpt_file.read_bytes() == b"PRECIOUS_NEURAL_WEIGHTS_DO_NOT_TOUCH"
    assert "my_private_conversations" in mem_file.read_text(encoding="utf-8")
    assert venv_marker.exists()


def test_download_exe_and_ota_endpoints(tmp_path: Path, monkeypatch) -> None:
    engine = NeverAIEngine(
        vault_dir=tmp_path / "ckpts",
        memory_path=tmp_path / "mem" / "evolvemem.json",
        telemetry_path=tmp_path / "logs" / "telemetry.jsonl",
        seq_len=64,
    )
    ota = OTAUpdater(app_root=tmp_path)
    monkeypatch.setattr(server_app, "_ENGINE", engine)
    monkeypatch.setattr(server_app, "_OTA", ota)

    client = TestClient(server_app.app)

    # 1. Native .exe endpoint
    r_exe = client.get("/download/NeverAI-Setup.exe")
    assert r_exe.status_code == 200
    assert 'filename="NeverAI-Setup.exe"' in r_exe.headers.get("content-disposition", "")
    assert r_exe.content[:2] == b"MZ"
    assert r_exe.content[SFX_ZIP_OFFSET : SFX_ZIP_OFFSET + 4] == b"PK\x03\x04"

    # 2. PowerShell 1-liner bootstrap
    r_ps = client.get("/api/ota/bootstrap.ps1")
    assert r_ps.status_code == 200
    assert "NeverAI" in r_ps.text

    # 3. OTA status & preferences
    r_st = client.get("/api/ota/status")
    assert r_st.status_code == 200
    assert r_st.json()["auto_update_on_boot"] is True

    r_pref = client.post("/api/ota/preferences", json={"auto_update_on_boot": False})
    assert r_pref.status_code == 200
    assert r_pref.json()["auto_update_on_boot"] is False

    # 4. Official W3C Android PWA endpoints (manifest, service worker, icons, mobile shell)
    r_man = client.get("/manifest.webmanifest")
    assert r_man.status_code == 200
    assert r_man.json()["display"] == "standalone"
    assert r_man.json()["version"] == "5.0.0"

    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert "SKIP_WAITING" in r_sw.text
    assert "NEVER_AI_SW_UPDATED" in r_sw.text
    assert "no-store" in r_sw.headers.get("cache-control", "")

    r_ver = client.get("/api/pwa/version")
    assert r_ver.status_code == 200
    assert r_ver.json()["version"] == "5.0.0"
    initial_build_id = r_ver.json()["build_id"]
    assert initial_build_id in r_sw.text

    # Simulate OTA commit advancement and verify /sw.js dynamically updates its BUILD stamp
    ota.state["local_commit"] = "9f8e7d6c5b4a32109f8e7d6c5b4a32109f8e7d6c"
    r_sw_after = client.get("/sw.js")
    assert "5.0.0-9f8e7d6" in r_sw_after.text

    assert client.get("/icon-192.png").status_code == 200
    assert client.get("/icon-512.png").status_code == 200
    r_mob = client.get("/mobile/index.html")
    assert r_mob.status_code == 200
    assert "initServiceWorkerSelfUpdate" in r_mob.text
    assert "runAutoOtaCycle" in r_mob.text

    # 5. Complete standalone NeverAI-PWA.zip package
    r_zip = client.get("/download/NeverAI-PWA.zip")
    assert r_zip.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r_zip.content), "r") as zf:
        names = set(zf.namelist())
        assert "NeverAI-PWA/index.html" in names
        assert "NeverAI-PWA/sw.js" in names
        assert "NeverAI-PWA/manifest.webmanifest" in names
        assert "NeverAI-PWA/icon-192.png" in names
        assert "NeverAI-PWA/icon-512.png" in names
        assert "NeverAI-PWA/LISEZMOI-PWA.md" in names


