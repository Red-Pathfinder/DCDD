# DCDD Execution & Workspace Directory

This directory contains the canonical evaluation engine, local dataset cache, and experiment configurations for the DCDD benchmark.

## Quick Start

From the repository root (cross-platform):
```bash
python reproduce.py
```

Or from within this directory:
```bash
python scripts/reproduce_paper.py
```

## Directory Overview

- **`scripts/reproduce_paper.py`**: Canonical reproduction engine that evaluates the benchmark and produces all manuscript tables, statistical tests, figures, and audit reports.
- **`dataset_cache/`**: Local cache of the published Hugging Face tabular dataset (manifests, VLM annotations, human validation records, and reliability study data) with SHA-256 checksums.
- **`results/reproduction/`**: Output directory for generated publication charts (`charts/`) and the markdown audit report (`DCDD_Paper_Reproduction_Report.md`).
- **`data_raw/` & `data_processed/`**: Benchmark staging directories.
- **`scripts/`**: Archival generation and annotation scripts from the initial data collection pipeline (see [scripts/README.md](scripts/README.md)).
