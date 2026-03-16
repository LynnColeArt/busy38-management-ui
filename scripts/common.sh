#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "scripts/common.sh is meant to be sourced, not executed directly." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

ensure_venv() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
}

ensure_management_requirements() {
  "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/backend/requirements.txt"
}

ensure_management_dev_requirements() {
  "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/backend/requirements-dev.txt"
}

resolve_busy_runtime_path() {
  if [[ -n "${BUSY_RUNTIME_PATH:-}" ]]; then
    printf '%s\n' "${BUSY_RUNTIME_PATH}"
    return 0
  fi

  if [[ -d "${REPO_ROOT}/../busy" ]]; then
    printf '%s\n' "${REPO_ROOT}/../busy"
    return 0
  fi

  return 1
}

require_busy_runtime() {
  local runtime_path
  if ! runtime_path="$(resolve_busy_runtime_path)"; then
    echo "Busy runtime path not found. Set BUSY_RUNTIME_PATH or place the Busy repo at ../busy." >&2
    exit 1
  fi

  export BUSY_RUNTIME_PATH="${runtime_path}"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_ROOT}:${BUSY_RUNTIME_PATH}:${PYTHONPATH}"
  else
    export PYTHONPATH="${REPO_ROOT}:${BUSY_RUNTIME_PATH}"
  fi
}

ensure_busy_runtime_support() {
  if ! "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
    "${PIP_BIN}" install PyYAML
  fi
}
