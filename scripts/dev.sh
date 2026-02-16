#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install -r "$ROOT/backend/requirements.txt"

echo "Serving API at http://127.0.0.1:8031"
echo "Open web/index.html manually or via any static server."
cd "$ROOT/backend"
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8031
