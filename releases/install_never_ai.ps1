# NEVER-AI (ChronoMorph-LM) — Installation & Mise a Jour OTA Automatique 1-Clic (Windows 10/11)
# Usage dans PowerShell :
#   irm "https://raw.githubusercontent.com/mouusieur/never-ai/arena/403f5d7f-never-ai/releases/install_never_ai.ps1" | iex

$ErrorActionPreference = "Stop"
$Target = Join-Path $env:LOCALAPPDATA "Programs\NeverAI"
Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  NEVER-AI v1.4 — Installation Souveraine + Mises a Jour OTA  " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# 1. Verification de Python 3.11+
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
    Write-Host "[1/4] Installation automatique de Python 3.11 via winget..." -ForegroundColor Yellow
    Start-Process -FilePath "winget" -ArgumentList "install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements" -Wait -NoNewWindow
    $sysPy = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (-not (Test-Path $sysPy)) { $sysPy = "python" }
}

# 2. Telechargement depuis GitHub HTTPS (sans ecraser checkpoints/ ni memory_store/)
Write-Host "[2/4] Synchronisation de la derniere version depuis GitHub (mouusieur/never-ai)..." -ForegroundColor Cyan
$ZipUrl = "https://codeload.github.com/mouusieur/never-ai/zip/arena/403f5d7f-never-ai"
$TmpZip = Join-Path $env:TEMP "never-ai-github-ota.zip"
$TmpExt = Join-Path $env:TEMP "never-ai-github-ota-ext"
Invoke-WebRequest -Uri $ZipUrl -OutFile $TmpZip -UseBasicParsing
Remove-Item $TmpExt -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $TmpZip -DestinationPath $TmpExt -Force
$Sub = Get-ChildItem -Path $TmpExt -Directory | Select-Object -First 1

New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $Sub.FullName "src") $Target
Copy-Item -Recurse -Force (Join-Path $Sub.FullName "assets") $Target
Copy-Item -Force (Join-Path $Sub.FullName "pyproject.toml") $Target
Copy-Item -Force (Join-Path $Sub.FullName "README.md") $Target

# 3. Creation de l'environnement virtuel isole (uniquement a la 1re installation)
$Venv = Join-Path $Target ".venv"
if (-not (Test-Path $Venv)) {
    Write-Host "[3/4] Premiere installation : configuration de ChronoMorph-LM & PyTorch CPU..." -ForegroundColor Cyan
    & $sysPy -m venv $Venv
    & (Join-Path $Venv "Scripts\pip.exe") install --upgrade pip
    & (Join-Path $Venv "Scripts\pip.exe") install --index-url https://download.pytorch.org/whl/cpu torch
    & (Join-Path $Venv "Scripts\pip.exe") install -e $Target pywebview
} else {
    Write-Host "[3/4] Environnement existant detecte (vos poids et discussions sont conserves)." -ForegroundColor Green
}

# 4. Raccourci Bureau Windows avec icone officielle never_ai.ico
Write-Host "[4/4] Creation du raccourci NEVER-AI sur le Bureau Windows..." -ForegroundColor Cyan
$Wsh = New-Object -ComObject WScript.Shell
$Lnk = $Wsh.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NEVER-AI.lnk"))
$Lnk.TargetPath = (Join-Path $Venv "Scripts\pythonw.exe")
$Lnk.Arguments = "`"$(Join-Path $Target 'src\never_ai\desktop\launcher.py')`""
$Lnk.WorkingDirectory = $Target
$Lnk.IconLocation = (Join-Path $Target "assets\brand\never_ai.ico")
$Lnk.Description = "NEVER-AI (ChronoMorph-LM) — Mises a jour OTA automatiques"
$Lnk.Save()

Write-Host ""
Write-Host ">>> NEVER-AI est pret ! Les futures mises a jour se feront automatiquement a distance." -ForegroundColor Green
Start-Process -FilePath $Lnk.TargetPath -ArgumentList $Lnk.Arguments -WorkingDirectory $Target
