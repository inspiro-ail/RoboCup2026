$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = ".\venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Run .\setup_rvc.ps1 first."
}
Write-Host "[+] Lorax AI: RVC + RMVPE" -ForegroundColor Cyan

Write-Host ""
Write-Host "[+] Запуск сервера Lorax AI..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

& $venvPython server.py
if ($LASTEXITCODE -ne 0) { throw "Server stopped with an error. Run .\setup_rvc.ps1 to check dependencies." }
