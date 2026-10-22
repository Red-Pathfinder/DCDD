#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

if [ -f "$ROOT_DIR/Drift/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/Drift/.venv/bin/python"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
fi

if ! "$PYTHON" -c "import numpy, pandas, scipy, statsmodels, matplotlib, huggingface_hub" 2>/dev/null; then
    echo "================================================================================"
    echo "DCDD Paper Reproduction Pipeline"
    echo "================================================================================"
    echo "Required Python dependencies are missing for: $PYTHON"
    echo ""
    echo "To set up the environment, run:"
    echo "    python3 -m venv .venv"
    echo "    source .venv/bin/activate"
    echo "    pip install -r requirements.txt"
    echo ""
    echo "Then run:"
    echo "    bash reproduce.sh"
    echo "================================================================================"
    exit 1
fi

exec "$PYTHON" "$ROOT_DIR/Drift/scripts/reproduce_paper.py" "$@"
