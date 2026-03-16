#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_venv
ensure_management_requirements
require_busy_runtime
ensure_busy_runtime_support

echo "Serving API at http://127.0.0.1:8031"
echo "Using Busy runtime at ${BUSY_RUNTIME_PATH}"
echo "Open http://127.0.0.1:8031/"
cd "${REPO_ROOT}/backend"
exec "${PYTHON_BIN}" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8031
