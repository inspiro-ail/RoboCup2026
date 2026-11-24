@echo off
title The Lorax AI Voice Studio - Public Sharing Tunnel
cd /d "%~dp0"
if exist "Lorax-AI" cd "Lorax-AI"

call venv\Scripts\python.exe share_tunnel.py
pause
