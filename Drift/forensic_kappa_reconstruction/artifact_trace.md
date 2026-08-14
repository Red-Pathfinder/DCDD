# Forensic Validation Audit: Artifact Trace

## 1. VLM Extraction Datasets
The human validation pipeline utilizes the **v2 schemas** for FLUX and SD3.5, and the `nanobanana_attributes_final.csv` for Nano Banana. 
- **FLUX**: `data_processed/01_flux_attributes_v2/flux_attributes_v2.csv` (381 rows)
- **SD 3.5**: `data_processed/02_sd35_attributes_v2/sd_attributes_v2.csv` (386 rows)
- **Nano Banana**: `data_processed/03_nanobanana_attributes/nanobanana_attributes_final.csv` (1,926 rows)

*Note: The primary analysis scripts (`05_flux_analysis.py` and `06_sd35_analysis.py`) actually point to the **v1** dataset folders (`01_flux_attributes`), establishing a disconnect between the data validated by humans and the data used for final demographic drift analysis.*

## 2. Human Validation Scripts
Three scripts are responsible for generating human validation data:

1. **`scripts/03_analysis/human_validation_of_sd_flux_non_amb.py`**
   - Validates FLUX and SD 3.5.
   - **Crucial Logic**: The script filters the dataset using `is_ambiguous(val)` and **only** presents images to the human where the VLM label contains "ambiguous", "unclear", or "indeterminate". Confident predictions are ignored.
   - Excludes the `sentiment` attribute entirely for FLUX and SD3.5 (defers to a "Math Merge" step).

2. **`scripts/03_analysis/08_human_audit.py`**
   - Validates Nano Banana.
   - Uses the same `is_ambiguous(val)` filter. Only ambiguous VLM predictions are presented to the human validator.

3. **`scripts/04_VLM_alignment_Flux/vlm_alignment_flux.py`**
   - Performs a limited audit on confident (non-ambiguous) VLM predictions.
   - Samples exactly 50 images, restricted exclusively to the `vision impairment` condition.
   - Writes to `vlm_alignment_audit.csv`.

## 3. Human Validation Artifacts (Outputs)
1. **`data_processed/human_overrides.csv`**
   - Size: 1,704 data rows.
   - Format: `master_id`, `attribute_name`, `corrected_value`
   - Purpose: Stores manual human corrections.
   - **Issues found**: The pipeline uses append-only CSV logic (`'a'` mode). While some session-level deduplication exists, it is possible to generate duplicate `(master_id, attribute)` overrides across multiple sessions.

2. **`data_processed/vlm_alignment_audit.csv`**
   - Size: 73 data rows.
   - Purpose: Records the 50-image confident prediction audit.
   - **Issues found**: The script lacks deduplication logic entirely. As a result, lines 2-23 are exact duplicates of lines 24-44. There are only ~51 unique audited items, not 73.

## 4. Final Validated Datasets (Pipeline Disconnect)
The pipeline orchestrator (`scripts/run_pipeline.py`) contains a `run_stage_human_validation` function, but it is an empty stub that only prints log messages. Furthermore, none of the primary analysis scripts (`05_flux_analysis.py`, `06_sd35_analysis.py`, `13_explicit_analysis_no_ambiguous.py`) ever load or apply `human_overrides.csv`. 

Therefore, **the final analysis artifacts are raw VLM predictions**. The human validation data exists solely in the overrides CSV and was never merged into the final datasets used for statistical testing in the repository scripts.
