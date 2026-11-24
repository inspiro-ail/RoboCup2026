# The Lorax AI Voice Studio - 1-Click Online Sharing
Set-Location -Path $PSScriptRoot

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "  * The Lorax AI Voice Studio - Public Online Tunnel *" -ForegroundColor Yellow
Write-Host "========================================================`n" -ForegroundColor Green

# 1. Start server if not running
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host "[1/2] Starting local AI server (FastAPI + RVC)..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c venv\Scripts\python.exe server.py" -WindowStyle Hidden
    Start-Sleep -Seconds 4
} else {
    Write-Host "[1/2] Server is already active on port 8000." -ForegroundColor Green
}

# 2. Launch cloudflared tunnel
Write-Host "[2/2] Connecting Cloudflare secure HTTPS tunnel..." -ForegroundColor Cyan
Write-Host "Your public HTTPS link will appear below (look for *.trycloudflare.com):`n" -ForegroundColor Yellow

& ".\bin\cloudflared.exe" tunnel --url http://127.0.0.1:8000

