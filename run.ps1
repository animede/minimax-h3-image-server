# MiniMax-H3-Image server for Windows (PowerShell). Default port 8633, listens on 127.0.0.1 only.
# Verified on Windows 11 with an RTX 3050 6GB / 16GB RAM (8gb profile, text encoder and transformer streamed from file).
#   .\run.ps1                       # http://127.0.0.1:8633/
#   $env:H3IMG_PORT=8634; .\run.ps1 # change the port
#   $env:H3IMG_HOST="0.0.0.0"; .\run.ps1   # expose on the LAN (no authentication)
Set-Location -Path $PSScriptRoot
$port = if ($env:H3IMG_PORT) { $env:H3IMG_PORT } else { "8633" }
$bind = if ($env:H3IMG_HOST) { $env:H3IMG_HOST } else { "127.0.0.1" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host $bind --port $port
