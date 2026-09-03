#!/usr/bin/env bash
# Create a Python 3.12+ virtual environment and install Phase I dependencies.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required; found {sys.version.split()[0]}")
PY
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -e "${REPO_ROOT}[dev,pipeline,viz]"
printf 'Environment ready. Activate with: source %q/bin/activate\n' "${VENV_PATH}"
