@echo off
rem Starts the MiniMax-H3-Image server without touching the PowerShell execution policy (unsigned .ps1 files are
rem blocked by default). Same environment variables as run.ps1 (H3IMG_PORT, H3IMG_HOST, ...).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
