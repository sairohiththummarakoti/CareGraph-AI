$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
        throw "Run .\scripts\setup.ps1 first."
    }
    & .\.venv\Scripts\python.exe app.py @args
    if ($LASTEXITCODE -ne 0) { throw "The CLI stopped with an error." }
} finally {
    Pop-Location
}
