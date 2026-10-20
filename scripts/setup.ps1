param([switch]$UsePip)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not $UsePip -and (Get-Command uv -ErrorAction SilentlyContinue)) {
        uv sync --python 3.12 --locked
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
        uv run python ingest.py
        if ($LASTEXITCODE -ne 0) { throw "Reference ingestion failed." }
    } else {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required. Install it and retry." }
        & .\.venv\Scripts\python.exe -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
        & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
        & .\.venv\Scripts\python.exe ingest.py
        if ($LASTEXITCODE -ne 0) { throw "Reference ingestion failed." }
    }
    Write-Host "Ready. Launch the workspace with .\scripts\launch-ui.ps1"
} finally {
    Pop-Location
}
