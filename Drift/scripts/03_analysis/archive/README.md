# Analysis Scripts - Archive

This directory contains older or experimental analysis scripts that are no longer part of the active pipeline.

## Current Active Scripts

The following scripts are the **current versions** and are executed by the pipeline orchestrator:

- **05_flux_analysis.py** — FLUX model statistical analysis and chart generation (primary)
- **06_sd35_analysis.py** — Stable Diffusion 3.5 statistical analysis and chart generation (primary)
- **03_nanobanana_analysis.py** — Nanobanana model statistical analysis and chart generation (primary)

## Archived Scripts

### `analysis_v2.py`
- **Purpose**: Early consolidated analysis combining all three models
- **Status**: Superseded by individual per-model analysis scripts
- **Reason archived**: The modular approach (separate analysis per model) is more maintainable and clearer
- **Date archived**: 2026-06-12

## Utility / Specialized Scripts (Still Active)

These scripts provide specialized analysis and validation functions, separate from the main pipeline:

- **08_human_audit.py** — Human validation and override management
- **10_erasure_analysis.py** — Specialized analysis for "erasure" bias patterns
- **11_intersectional_analysis.py** — Intersectional demographic analysis
- **13_explicit_analysis_no_ambiguous.py** — Analysis filtering ambiguous predictions
- **human_validation_of_nanobanana_ambiguous_img.py** — Nanobanana-specific validation utility
- **human_validation_of_sd_flux_non_amb.py** — FLUX/SD3.5 validation utility
- **03_nanobanana_charts.py** — Chart generation helper for Nanobanana

These are run on-demand for specific analyses, not as part of the main pipeline.

## How to Re-run Analysis

Use the pipeline orchestrator to re-run all analyses:

```bash
python scripts/run_pipeline.py --stages analysis
```

Or run a specific model's analysis:

```bash
python scripts/run_pipeline.py --stages analysis --models flux
```

To manually run a specific analysis script:

```bash
python 03_analysis/05_flux_analysis.py
```

## Note on Versioning

The pipeline now uses a **config-driven approach** via `pipeline_config.yaml`. All parameters (models, stages, output paths) are centralized in the config. If you need to revert to an archived script, restore it from this archive directory and update `pipeline_config.yaml` to reference it.

## Future Archival

New versions of analysis scripts should follow this pattern:
1. Create new script with descriptive name or version number
2. Update `pipeline_config.yaml` to reference it
3. Once stable, move old version to this archive/ directory
4. Update this README
