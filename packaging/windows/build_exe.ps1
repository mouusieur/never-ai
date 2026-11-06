# Build script PowerShell 1-clic pour générer dist\NeverAI\NeverAI.exe sur Windows 10/11
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor White
Write-Host "  NEVER-AI DESKTOP (.EXE) — WINDOWS BUILD PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor White

if (-not (Test-Path ".venv_win")) {
    python -m venv .venv_win
}

& .\.venv_win\Scripts\pip.exe install --upgrade pip
& .\.venv_win\Scripts\pip.exe install -e . pywebview pyinstaller

# Générer l'icône multi-résolution never_ai.ico si absente
& .\.venv_win\Scripts\python.exe scripts\generate_brand_assets.py

# Compiler l'application Windows sans console noire (--noconsole via spec)
& .\.venv_win\Scripts\pyinstaller.exe --noconfirm --clean packaging\windows\never_ai_desktop.spec

# Copier le dernier cerveau entraîné s'il est présent localement
if (Test-Path "checkpoints\latest") {
    New-Item -ItemType Directory -Force -Path "dist\NeverAI\never_ai_data\checkpoints" | Out-Null
    Copy-Item -Recurse -Force "checkpoints\latest" "dist\NeverAI\never_ai_data\checkpoints\latest"
    Write-Host "[OK] Cerveau Sprout-MoE intégré dans dist\NeverAI\never_ai_data\checkpoints\latest" -ForegroundColor Green
}

Write-Host "`n[TERMINE] Lancez votre application : dist\NeverAI\NeverAI.exe" -ForegroundColor Green
