#!/usr/bin/env bash
set -e
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Starting on http://localhost:8000  (Ctrl+C to stop)"
uvicorn app.main:app --reload --port 8000