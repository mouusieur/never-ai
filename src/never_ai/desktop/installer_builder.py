"""On-the-fly Windows Graphical Installer Bundle Builder (`NeverAI-Installer-Windows-x64.zip`)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

from never_ai.updater.ota import detect_git_head_info


_INSTALLER_BAT = r"""@echo off
title NEVER-AI — Assistant d'Installation Windows
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0setup_wizard.ps1"
"""

_SETUP_WIZARD_PS1 = r"""# NEVER-AI — Native Dark Windows Graphical Setup Wizard (.NET WinForms)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultTarget = Join-Path $env:LOCALAPPDATA "Programs\NeverAI"

$form = New-Object System.Windows.Forms.Form
$form.Text = "NEVER-AI — Installation Souveraine (Windows x64)"
$form.Size = New-Object System.Drawing.Size(600, 460)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.BackColor = [System.Drawing.Color]::FromArgb(0, 0, 0)
$form.ForeColor = [System.Drawing.Color]::FromArgb(231, 233, 234)

$icoPath = Join-Path $BundleRoot "never-ai-src\assets\brand\never_ai.ico"
if (Test-Path $icoPath) {
    $form.Icon = New-Object System.Drawing.Icon($icoPath)
}

$lblTitle = New-Object System.Windows.Forms.Label
$lblTitle.Text = "NEVER-AI"
$lblTitle.Font = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
$lblTitle.AutoSize = $true
$lblTitle.Location = New-Object System.Drawing.Point(28, 22)
$form.Controls.Add($lblTitle)

$lblSub = New-Object System.Windows.Forms.Label
$lblSub.Text = "La pensee libre. Sans fin.  —  ChronoMorph-LM Desktop"
$lblSub.Font = New-Object System.Drawing.Font("Segoe UI", 10)
$lblSub.ForeColor = [System.Drawing.Color]::FromArgb(113, 118, 123)
$lblSub.AutoSize = $true
$lblSub.Location = New-Object System.Drawing.Point(32, 68)
$form.Controls.Add($lblSub)

$lblDir = New-Object System.Windows.Forms.Label
$lblDir.Text = "Dossier d'installation :"
$lblDir.Font = New-Object System.Drawing.Font("Segoe UI", 9.5, [System.Drawing.FontStyle]::Bold)
$lblDir.AutoSize = $true
$lblDir.Location = New-Object System.Drawing.Point(32, 108)
$form.Controls.Add($lblDir)

$txtDir = New-Object System.Windows.Forms.TextBox
$txtDir.Text = $DefaultTarget
$txtDir.Size = New-Object System.Drawing.Size(520, 28)
$txtDir.Location = New-Object System.Drawing.Point(32, 132)
$txtDir.BackColor = [System.Drawing.Color]::FromArgb(22, 24, 28)
$txtDir.ForeColor = [System.Drawing.Color]::FromArgb(231, 233, 234)
$txtDir.BorderStyle = "FixedSingle"
$form.Controls.Add($txtDir)

$chkDesktop = New-Object System.Windows.Forms.CheckBox
$chkDesktop.Text = "Creer un raccourci NEVER-AI sur le Bureau (avec icone officielle)"
$chkDesktop.Checked = $true
$chkDesktop.AutoSize = $true
$chkDesktop.Location = New-Object System.Drawing.Point(32, 174)
$form.Controls.Add($chkDesktop)

$chkStartMenu = New-Object System.Windows.Forms.CheckBox
$chkStartMenu.Text = "Ajouter NEVER-AI au Menu Demarrer de Windows"
$chkStartMenu.Checked = $true
$chkStartMenu.AutoSize = $true
$chkStartMenu.Location = New-Object System.Drawing.Point(32, 202)
$form.Controls.Add($chkStartMenu)

$chkBuildExe = New-Object System.Windows.Forms.CheckBox
$chkBuildExe.Text = "Compiler egalement le binaire autonome NeverAI.exe (PyInstaller)"
$chkBuildExe.Checked = $false
$chkBuildExe.AutoSize = $true
$chkBuildExe.Location = New-Object System.Drawing.Point(32, 230)
$form.Controls.Add($chkBuildExe)

$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Multiline = $true
$txtLog.ReadOnly = $true
$txtLog.ScrollBars = "Vertical"
$txtLog.Size = New-Object System.Drawing.Size(520, 96)
$txtLog.Location = New-Object System.Drawing.Point(32, 264)
$txtLog.BackColor = [System.Drawing.Color]::FromArgb(12, 14, 18)
$txtLog.ForeColor = [System.Drawing.Color]::FromArgb(110, 231, 183)
$txtLog.BorderStyle = "FixedSingle"
$txtLog.Font = New-Object System.Drawing.Font("Consolas", 8.5)
$txtLog.Text = "Pret a installer NEVER-AI en local sur cet appareil.`r`nMises a jour a distance (OTA) incluses : vous n'aurez plus jamais a reinstaller !`r`n"
$form.Controls.Add($txtLog)

$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = "Installer et Lancer NEVER-AI"
$btnInstall.Size = New-Object System.Drawing.Size(260, 40)
$btnInstall.Location = New-Object System.Drawing.Point(292, 374)
$btnInstall.FlatStyle = "Flat"
$btnInstall.BackColor = [System.Drawing.Color]::FromArgb(239, 243, 244)
$btnInstall.ForeColor = [System.Drawing.Color]::FromArgb(15, 20, 25)
$btnInstall.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($btnInstall)

function Log-Step([string]$msg) {
    $txtLog.AppendText("$msg`r`n")
    [System.Windows.Forms.Application]::DoEvents()
}

$btnInstall.Add_Click({
    $btnInstall.Enabled = $false
    try {
        $Target = $txtDir.Text.Trim()
        Log-Step "[1/5] Verification de Python 3.11+..."
        $sysPy = $null
        foreach ($cand in @("python", "py", "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe")) {
            try {
                $verOut = & $cand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($LASTEXITCODE -eq 0 -and $verOut -match "^3\.(1[1-9]|[2-9]\d)") {
                    $sysPy = $cand
                    break
                }
            } catch {}
        }
        if (-not $sysPy) {
            Log-Step "Installation automatique de Python 3.11 via winget..."
            Start-Process -FilePath "winget" -ArgumentList "install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements" -Wait -NoNewWindow
            $sysPy = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
            if (-not (Test-Path $sysPy)) { $sysPy = "python" }
        }

        Log-Step "[2/5] Copie des fichiers sources et icone dans $Target..."
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
        Copy-Item -Recurse -Force (Join-Path $BundleRoot "never-ai-src\*") $Target

        Log-Step "[3/5] Creation de l'environnement virtuel isole (.venv)..."
        $venvPath = Join-Path $Target ".venv"
        if (-not (Test-Path $venvPath)) {
            & $sysPy -m venv $venvPath
        }
        $pipExe = Join-Path $venvPath "Scripts\pip.exe"
        $pywExe = Join-Path $venvPath "Scripts\pythonw.exe"
        $pyExe  = Join-Path $venvPath "Scripts\python.exe"

        Log-Step "[4/5] Installation de ChronoMorph-LM, FastAPI & WebView2..."
        & $pipExe install --upgrade pip | Out-Null
        & $pipExe install --index-url https://download.pytorch.org/whl/cpu torch | Out-Null
        & $pipExe install -e $Target pywebview | Out-Null

        # Create silent zero-console launcher VBS + optional PyInstaller .exe
        $launcherPy = Join-Path $Target "src\never_ai\desktop\launcher.py"
        $launchTarget = $pywExe
        $launchArgs = "`"$launcherPy`""

        if ($chkBuildExe.Checked) {
            Log-Step "[4b/5] Compilation du binaire natif NeverAI.exe..."
            & $pipExe install pyinstaller | Out-Null
            Push-Location $Target
            & (Join-Path $venvPath "Scripts\pyinstaller.exe") --noconfirm --clean packaging\windows\never_ai_desktop.spec | Out-Null
            Pop-Location
            $builtExe = Join-Path $Target "dist\NeverAI\NeverAI.exe"
            if (Test-Path $builtExe) {
                $launchTarget = $builtExe
                $launchArgs = ""
            }
        }

        Log-Step "[5/5] Creation des raccourcis Windows avec icone never_ai.ico..."
        $wsh = New-Object -ComObject WScript.Shell
        $targetIcon = Join-Path $Target "assets\brand\never_ai.ico"

        if ($chkDesktop.Checked) {
            $deskLnk = $wsh.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NEVER-AI.lnk"))
            $deskLnk.TargetPath = $launchTarget
            $deskLnk.Arguments = $launchArgs
            $deskLnk.WorkingDirectory = $Target
            $deskLnk.IconLocation = $targetIcon
            $deskLnk.Description = "NEVER-AI (ChronoMorph-LM)"
            $deskLnk.Save()
        }

        if ($chkStartMenu.Checked) {
            $smDir = Join-Path ([Environment]::GetFolderPath("Programs")) "NEVER-AI"
            New-Item -ItemType Directory -Force -Path $smDir | Out-Null
            $smLnk = $wsh.CreateShortcut((Join-Path $smDir "NEVER-AI.lnk"))
            $smLnk.TargetPath = $launchTarget
            $smLnk.Arguments = $launchArgs
            $smLnk.WorkingDirectory = $Target
            $smLnk.IconLocation = $targetIcon
            $smLnk.Save()
        }

        Log-Step "Installation terminee ! Lancement de NEVER-AI..."
        if ($launchArgs -eq "") {
            Start-Process -FilePath $launchTarget -WorkingDirectory $Target
        } else {
            Start-Process -FilePath $launchTarget -ArgumentList $launchArgs -WorkingDirectory $Target
        }
        $form.Close()
    } catch {
        Log-Step "ERREUR : $($_.Exception.Message)"
        $btnInstall.Enabled = $true
    }
})

[void]$form.ShowDialog()
"""

_README_FR = """========================================================================
  NEVER-AI (ChronoMorph-LM) — INSTALLATEUR WINDOWS 10 / 11 (x64)
========================================================================

INSTALLATION EN 2 SECONDES :
1. Extrayez cette archive ZIP dans le dossier de votre choix.
2. Double-cliquez sur :  Installer-NeverAI.bat

L'assistant graphique noir (Monolith Noir) s'ouvre automatiquement :
- Installe NEVER-AI en local dans %LOCALAPPDATA%\\Programs\\NeverAI
- Cree le raccourci Bureau et Menu Demarrer avec l'icone officielle NEVER-AI
- Lance la fenetre native sombre sans console noire (WebView2 / NeverAI.exe)
- Option incluse : compiler directement le binaire autonome NeverAI.exe !
========================================================================
"""


def build_windows_installer_zip(repo_root: str | Path = ".") -> bytes:
    """Package the graphical Windows installer + full Never-AI source & brand kit into a ZIP."""
    root = Path(repo_root).resolve()
    buf = io.BytesIO()

    include_files: list[Path] = []
    for rel in ("pyproject.toml", "README.md", "DIRECTIVES.md"):
        p = root / rel
        if p.exists():
            include_files.append(p)

    for folder in ("src", "assets", "scripts", "packaging", "docs"):
        d = root / folder
        if d.exists():
            for fp in d.rglob("*"):
                if fp.is_file() and "__pycache__" not in fp.parts and not fp.name.endswith(".pyc"):
                    include_files.append(fp)

    # Also bundle tokenizer & memory store if present (lightweight < 400 KB)
    tok_p = root / "checkpoints" / "latest" / "tokenizer.json"
    cfg_p = root / "checkpoints" / "latest" / "config.json"

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Installer-NeverAI.bat", _INSTALLER_BAT.replace("\n", "\r\n"))
        zf.writestr("setup_wizard.ps1", _SETUP_WIZARD_PS1.replace("\n", "\r\n"))
        zf.writestr("LISEZ-MOI.txt", _README_FR.replace("\n", "\r\n"))

        sha, br = detect_git_head_info(root)
        ota_stamp = json.dumps({"commit": sha, "branch": br}, indent=2)
        zf.writestr("never-ai-src/ota_manifest.json", ota_stamp)

        for fp in include_files:
            arcname = f"never-ai-src/{fp.relative_to(root).as_posix()}"
            zf.write(fp, arcname=arcname)

        if tok_p.exists():
            zf.write(tok_p, arcname="never-ai-src/memory_store/seed_tokenizer.json")
        if cfg_p.exists():
            zf.write(cfg_p, arcname="never-ai-src/memory_store/seed_config.json")

    return buf.getvalue()
