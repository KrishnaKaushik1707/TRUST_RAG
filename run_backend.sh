#!/usr/bin/env bash
# Quick launcher for TrustRAG backend
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/backend"
PYTHONPATH=. "$ROOT_DIR/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8000 --reload
