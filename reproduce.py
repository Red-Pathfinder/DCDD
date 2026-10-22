#!/usr/bin/env python3
"""
Canonical Cross-Platform Reproduction Entry Point
Disability-Conditioned Demographic Drift (DCDD) — WACV 2027

This script provides an OS-independent entry point (Windows, macOS, Linux)
for reproducing all empirical tables, figures, and statistical tests reported in:
"Disability-Conditioned Demographic Drift: Intersectional Stereotyping and
Algorithmic Erasure in Text-to-Image Generative Models" (WACV 2027).

Usage:
    python reproduce.py                  # Standard tabular reproduction (default)
    python reproduce.py --mode full      # Download and verify complete image set (~1.6 GB)
    python reproduce.py --download       # Force fresh retrieval from Hugging Face
"""

import os
import sys
import subprocess
from pathlib import Path

MIN_PYTHON = (3, 10)
REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"),
    ("huggingface_hub", "huggingface_hub"),
]


def find_local_venv(root_dir: Path) -> Path | None:
    """Detect local virtual environments across Windows, Linux, and macOS."""
    candidates = [
        root_dir / ".venv" / "bin" / "python",
        root_dir / ".venv" / "Scripts" / "python.exe",
        root_dir / "Drift" / ".venv" / "bin" / "python",
        root_dir / "Drift" / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def get_missing_packages() -> list[str]:
    """Check availability of required runtime dependencies."""
    missing = []
    for import_name, pkg_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    return missing


def main() -> None:
    """Validate environment and execute canonical reproduction engine."""
    root_dir = Path(__file__).resolve().parent

    # Enforce minimum Python version
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or higher is required. "
            f"Detected Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    missing = get_missing_packages()

    # If current environment is missing packages, check if a local .venv exists with them
    if missing and "DCDD_VENV_ATTEMPTED" not in os.environ:
        venv_python = find_local_venv(root_dir)
        if venv_python and venv_python != Path(sys.executable).resolve():
            env = os.environ.copy()
            env["DCDD_VENV_ATTEMPTED"] = "1"
            result = subprocess.run([str(venv_python), str(__file__)] + sys.argv[1:], env=env)
            sys.exit(result.returncode)

    if missing:
        msg = [
            "=" * 80,
            "DCDD Reproduction Pipeline — Environment Setup Required",
            "=" * 80,
            f"Missing required packages: {', '.join(missing)}\n",
            "To install dependencies in a virtual environment, run:",
            "    python3 -m venv .venv",
            "    source .venv/bin/activate       # On Windows: .venv\\Scripts\\activate",
            "    pip install -r requirements.txt\n",
            "Alternatively, if using Conda:",
            "    conda env create -f environment.yml",
            "    conda activate dcdd\n",
            "Then execute:",
            "    python reproduce.py",
            "=" * 80,
        ]
        sys.exit("\n".join(msg))

    engine_path = root_dir / "Drift" / "scripts" / "reproduce_paper.py"
    if not engine_path.is_file():
        sys.exit(f"Error: Reproduction engine not found at expected path: {engine_path}")

    cmd = [sys.executable, str(engine_path)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
