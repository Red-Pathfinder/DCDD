# DCDD Pipeline Execution Guide

Complete step-by-step guide for running the Drift data generation, extraction, and analysis pipeline.

## Quick Start

For a complete pipeline run from scratch (all stages, all models):

```bash
cd Drift
python scripts/run_pipeline.py
```

For a specific model or stage:

```bash
# Run only FLUX analysis
python scripts/run_pipeline.py --models flux --stages analysis

# Run extraction and analysis for all models
python scripts/run_pipeline.py --stages attribute_extraction analysis

# Dry run (see what would happen without executing)
python scripts/run_pipeline.py --dry-run
```

## Pipeline Overview

The pipeline consists of 5 sequential stages:

```
Stage 1: Image Generation
    ↓ (generates images with/without disability keywords)
Stage 2: Attribute Extraction
    ↓ (VLM extracts demographics from images)
Stage 3: Compliance Cleanup
    ↓ (removes teen data, applies filters)
Stage 4: Human Validation
    ↓ (applies manual corrections)
Stage 5: Analysis & Charts
    ↓ (statistical testing and visualization)
```

## Stage Details

### Stage 1: Image Generation

**What it does:**
- Generates images using FLUX, SD3.5, and Nanobanana
- Creates two versions of each: "neutral" (baseline) and "conditioned" (disability keyword added)
- Saves to `data_raw/flux_pipeline/`, `data_raw/sd35_pipeline/`, `data_raw/nanobanana_2_pipeline/`

**Scripts:**
- `01_generate_flux.py`
- `02_generate_sd35.py`
- `03_nanabanana.py`

**Time estimate:** 2-4 hours (per model, depending on API rate limits)

**Configuration (pipeline_config.yaml):**
```yaml
stages:
  - id: 1
    name: "image_generation"
    skip_if_exists: true  # Skip if images already exist
```

**Manual override:**
```bash
python scripts/run_pipeline.py --stages image_generation --models flux
```

---

### Stage 2: Attribute Extraction

**What it does:**
- Uses Vision Language Model (gpt-4-mini) to extract demographic attributes
- Processes all generated images
- Outputs `attributes.csv` with 12 demographic columns
- Saves to `data_processed/{model}_attributes/`

**Scripts:**
- `02_attribute_extraction/01_flux_attribute.py`
- `02_attribute_extraction/02_sd35_attribute.py`
- `02_attribute_extraction/03_nanobanana_attribute.py`

**Time estimate:** 1-2 hours (per model, API calls can be slow)

**Configuration (pipeline_config.yaml):**
```yaml
stages:
  - id: 2
    name: "attribute_extraction"
    vlm_model: "gpt-4-mini"
    skip_if_exists: true
```

**Manual override:**
```bash
python scripts/run_pipeline.py --stages attribute_extraction --models sd35,nanobanana
```

**Output schema:**
- `image_id`, `image_type` (neutral/conditioned)
- 12 demographic attributes (gender, race, age, sentiment, etc.)

---

### Stage 3: Compliance Cleanup

**What it does:**
- Applies legal/ethical compliance filters
- Removes teen data (age 13-17) from all datasets
- Removes other sensitive categories as needed
- Updates attributes CSVs with filtered data

**Configuration (pipeline_config.yaml):**
```yaml
stages:
  - id: 3
    name: "compliance_cleanup"
    filters:
      - name: "remove_teen_data"
        enabled: true
        condition: "age_bin != 'teen (13-17)'"
        reason: "GDPR/legal compliance"
```

**Script:** `00_purge_flux_teens.py` (called automatically)

**Time estimate:** < 5 minutes

---

### Stage 4: Human Validation

**What it does:**
- Loads manual corrections from `human_overrides.csv`
- Applies corrections to VLM predictions
- Replaces incorrect attribute values with human-validated ones
- Updates attributes CSVs with corrections

**Override CSV format:**
```
image_id,attribute,original_prediction,override_value,reason,validator_name,validation_date
001_flux_neutral,gender_presentation,male,female,clear error - appears female,Alice,2026-05-20
```

**Time estimate:** < 5 minutes

---

### Stage 5: Analysis & Charts

**What it does:**
- Performs statistical analysis (chi-squared tests)
- Generates demographic shift charts
- Creates master comparison Excel file: `DCDD_Final_Master_Analysis.xlsx`
- Outputs individual model analysis CSVs and PNGs

**Scripts:**
- `03_analysis/05_flux_analysis.py`
- `03_analysis/06_sd35_analysis.py`
- `03_analysis/03_nanobanana_analysis.py`

**Output files:**
- `results/charts/{model}_*.png` — individual charts
- `results/statistical_summary/{model}_statistical_summary.csv` — stats
- `results/DCDD_Final_Master_Analysis.xlsx` — master comparison

**Time estimate:** 10-20 minutes (per model)

**Manual override:**
```bash
python scripts/run_pipeline.py --stages analysis
```

---

## Running Specific Scenarios

### Scenario 1: Add a New Model

1. Add model to `pipeline_config.yaml`:
```yaml
models:
  - name: "my_new_model"
    gen_script: "01_image_generation/04_generate_new_model.py"
    extract_script: "02_attribute_extraction/04_new_model_attribute.py"
    analysis_script: "03_analysis/04_new_model_analysis.py"
    data_folder: "04_new_model_attributes"
    output_prefix: "new_model"
```

2. Run pipeline for that model:
```bash
python scripts/run_pipeline.py --models my_new_model
```

### Scenario 2: Re-run Analysis Only

If you've made changes to analysis scripts or want to re-generate charts:

```bash
python scripts/run_pipeline.py --stages analysis
```

### Scenario 3: Re-run With Alternate Data Version

If you want to use the alternate `01_flux_attributes_v2` instead of the primary version:

1. Edit `pipeline_config.yaml`:
```yaml
models:
  - name: "flux"
    data_folder: "01_flux_attributes_v2"  # Changed from 01_flux_attributes
```

2. Run analysis:
```bash
python scripts/run_pipeline.py --stages analysis --models flux
```

### Scenario 4: Validate Pipeline State

Before running analysis, validate that all outputs are consistent:

```bash
python scripts/validate_stage_outputs.py
```

This checks:
- All attributes CSVs have correct columns and structure
- Image counts match between raw images and attribute rows
- No teen data (compliance check)
- Human overrides are valid

---

## Configuration Reference

### pipeline_config.yaml

Main configuration file at `Drift/pipeline_config.yaml`:

```yaml
# Project metadata
project:
  name: "DCDD - Disability and Demographic Drift"

# Models to process
models:
  - name: "flux"
    data_folder: "01_flux_attributes"
    ...

# Pipeline stages
stages:
  - id: 1
    name: "image_generation"
    enabled: true
    skip_if_exists: true

# Analysis parameters
analysis:
  vlm_model: "gpt-4-mini"
  dpi: 300
  significance_level: 0.05

# Execution settings
execution:
  parallel_models: false  # Set to true for parallel processing (experimental)
  stop_on_error: true
```

Edit this file to:
- Enable/disable specific stages
- Change VLM model
- Add new models
- Modify analysis parameters
- Toggle parallel execution

---

## Logging & Debugging

Pipeline logs are saved to `Drift/logs/pipeline_run_YYYY-MM-DD_HH-MM-SS.log`

To follow the log in real-time:
```bash
tail -f "Drift/logs/pipeline_run_*.log"
```

To increase logging verbosity, edit `pipeline_config.yaml`:
```yaml
logging:
  level: "DEBUG"  # More verbose output
```

---

## Troubleshooting

### Issue: Image count mismatch

**Problem:** `Validation error: 150 images vs 100 attribute rows`

**Solution:**
1. Check if extraction script failed partway through
2. Re-run attribute extraction: `python scripts/run_pipeline.py --stages attribute_extraction`
3. Check logs: `tail Drift/logs/pipeline_run_*.log`

### Issue: VLM API rate limit

**Problem:** `OpenAI API rate limit exceeded`

**Solution:**
1. Add delay between requests (edit extraction script)
2. Run stage again later when rate limit resets
3. Use a different VLM model (edit `pipeline_config.yaml`)

### Issue: Teen data not removed

**Problem:** Validation says "Teen data found"

**Solution:**
1. Ensure compliance_cleanup stage is enabled
2. Check that `human_overrides.csv` doesn't have teen data
3. Re-run cleanup: `python scripts/run_pipeline.py --stages compliance_cleanup`

---

## Output Files

After a complete pipeline run, you'll have:

```
Drift/
├── data_raw/
│   ├── flux_pipeline/          # Generated images
│   ├── sd35_pipeline/
│   └── nanobanana_2_pipeline/
├── data_processed/
│   ├── 01_flux_attributes/     # Extracted attributes
│   │   └── attributes.csv
│   ├── 02_sd35_attributes/
│   │   └── attributes.csv
│   └── human_overrides.csv     # Manual corrections
└── ../results/
    ├── charts/
    │   ├── flux_*.png
    │   ├── sd35_*.png
    │   └── nanobanana_*.png
    ├── statistical_summary/
    │   ├── flux_statistical_summary.csv
    │   ├── sd35_statistical_summary.csv
    │   └── nanobanana_statistical_summary.csv
    └── DCDD_Final_Master_Analysis.xlsx
```

---

## Next Steps

1. **Run validator first:** `python scripts/validate_stage_outputs.py`
2. **Review current data:** Check `data_processed/README.md` for schema
3. **Run full pipeline:** `python scripts/run_pipeline.py`
4. **Check results:** Open `results/DCDD_Final_Master_Analysis.xlsx`

For questions or issues, see:
- `Drift/README.md` — project overview
- `Drift/scripts/README.md` — script organization
- `Drift/data_processed/README.md` — data format reference
