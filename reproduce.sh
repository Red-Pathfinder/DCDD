#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if [ -f "$ROOT_DIR/Drift/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/Drift/.venv/bin/python"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
fi

exec "$PYTHON" "$ROOT_DIR/Drift/scripts/reproduce_paper.py" "$@"
