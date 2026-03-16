#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_venv
ensure_management_requirements
ensure_management_dev_requirements
require_busy_runtime
ensure_busy_runtime_support

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m pytest "${@:-tests}"
