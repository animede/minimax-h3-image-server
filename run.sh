#!/usr/bin/env bash
# MiniMax-H3-Image server (default port 8633). Paths resolve relative to this directory.
# 既定ではローカルのみ待ち受ける。LAN に公開するときは H3IMG_HOST=0.0.0.0 を指定する。
set -euo pipefail
cd "$(dirname "$0")"
PORT="${H3IMG_PORT:-8633}"
HOST="${H3IMG_HOST:-127.0.0.1}"
exec .venv/bin/python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
