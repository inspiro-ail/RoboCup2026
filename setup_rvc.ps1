$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & python -m venv (Join-Path $PSScriptRoot "venv")
    if ($LASTEXITCODE -ne 0) { throw "Cannot create Python environment." }
}
& $venvPython setup_rvc.py
if ($LASTEXITCODE -ne 0) { throw "RVC setup failed. See the error above." }
