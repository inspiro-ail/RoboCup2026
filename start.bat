@echo off
cd /d "%~dp0"

echo Starting Lorax AI Voice Studio - RVC + RMVPE...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
