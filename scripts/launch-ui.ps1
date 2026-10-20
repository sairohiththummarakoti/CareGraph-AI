$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
        throw "Run .\scripts\setup.ps1 first."
    }
    Write-Host "CareGraph AI: http://127.0.0.1:8501 (Ctrl+C to stop)"
    & .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
    if ($LASTEXITCODE -ne 0) { throw "The Streamlit server stopped with an error." }
} finally {
    Pop-Location
}
