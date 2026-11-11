#!/usr/bin/env bash
# MiniMax-H3-Image server (default port 8633). Paths resolve relative to this directory.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${H3IMG_PORT:-8633}"
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
