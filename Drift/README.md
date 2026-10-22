# DCDD Pipeline & Execution Directory

This directory contains the reproduction pipeline, dataset cache, and experiment scripts for the DCDD benchmark.

## Quick Start

From the repository root:
```bash
bash reproduce.sh
```

Or from inside this directory:
```bash
python scripts/reproduce_paper.py
```

## Directory Overview

- **`scripts/reproduce_paper.py`**: Canonical reproduction script that computes all paper tables, statistical tests, figures, and audit reports.
- **`dataset_cache/`**: Local cache of the released Hugging Face tabular dataset (manifests, VLM annotations, human validation records, and reliability study data) with SHA-256 checksums.
- **`results/reproduction/`**: Output directory for generated publication charts (`charts/`) and the markdown audit report (`DCDD_Paper_Reproduction_Report.md`).
- **`data_raw/` & `data_processed/`**: Local data staging directories.
- **`scripts/`**: Archival generation and annotation scripts from the initial benchmark construction (see [scripts/README.md](scripts/README.md)).
