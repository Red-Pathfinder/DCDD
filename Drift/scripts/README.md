# Scripts Directory - Organization & Reference

This directory contains reproduction scripts, statistical pipelines, and historical generation/annotation utilities.

## ★ Official Paper Reproduction (WACV 2027)

To reproduce the paper's statistical results, reliability figures, and metrics using the published Hugging Face dataset:

```bash
# From Drift/ directory:
bash reproduce.sh                 # Default: tabular mode (~1.2 MB download, fast)
bash reproduce.sh --mode full      # Full mode: downloads all 2,472 images (~1.6 GB)

# Or directly via Python:
python scripts/reproduce_paper.py --mode tabular
```

---

## Historical Data Generation & VLM Extraction

> **Note for Researchers:** The scripts below (`run_pipeline.py`, `01_image_generation/`, `02_attribute_extraction/`) represent the original multi-stage generation and annotation pipeline used to produce the raw benchmark data. They require paid third-party API credentials (fal.ai, Google Gemini API) and operate on raw local image directories. They are retained for archival completeness and transparency, but are **not** needed to reproduce the published findings.

### Historical Pipeline Structure

```
scripts/
├── reproduce_paper.py                ★ OFFICIAL REPRODUCTION ENTRY POINT (WACV 2027)
├── run_pipeline.py                   [Historical] Generation & annotation orchestrator
├── validate_stage_outputs.py         [Historical] Validation before analysis
│
├── 00_purge_flux_teens.py            Data cleanup (called by pipeline)
│
├── 01_image_generation/              Stage 1: Generate images (Requires API keys)
│   ├── 01_generate_flux.py
│   ├── 02_generate_sd35.py
│   ├── 03_nanabanana.py
│   └── 04_rescue_batch.py
│
├── 02_attribute_extraction/          Stage 2: Extract demographics (Requires VLM API keys)
│   ├── 01_flux_attribute.py
│   ├── 01_flux_v2_attribute.py
│   ├── 02_sd35_attribute.py
│   ├── 02_sd35_v2_attribute.py
│   ├── 03_nanobanana_attribute.py
│   └── 09_profession_erasure_batch.py
│
├── 03_analysis/                      Stage 5: Early model-specific analysis scripts
│   ├── 05_flux_analysis.py           [Historical FLUX exploratory script]
│   ├── 06_sd35_analysis.py           [Historical SD3.5 exploratory script]
│   ├── 03_nanobanana_analysis.py     [Historical Nanobanana exploratory script]
│   │
│   ├── archive/                      Deprecated/old scripts
│   │   ├── analysis_v2.py            (old consolidated version)
│   │   └── README.md
│   │
│   └── [Utility scripts]
│       ├── 03_nanobanana_charts.py   Chart generation utility
│       ├── 08_human_audit.py         Human validation utility
│       ├── 10_erasure_analysis.py    Erasure bias analysis
│       ├── 11_intersectional_analysis.py    Intersectional analysis
│       ├── 13_explicit_analysis_no_ambiguous.py    Variant analysis
│       ├── human_validation_of_nanobanana_ambiguous_img.py    Validation
│       └── human_validation_of_sd_flux_non_amb.py    Validation
│
├── 04_VLM_alignment_Flux/            Special: VLM alignment testing
├── mitigation/                        Special: Bias mitigation experiments
├── plotting/                          Special: Visualization utilities
│
└── README.md                         This file
```

## Stage Breakdown

### Stage 1: Image Generation

**Active Scripts:**
- `01_image_generation/01_generate_flux.py` — Generate images using FLUX API
- `01_image_generation/02_generate_sd35.py` — Generate images using Stable Diffusion 3.5
- `01_image_generation/03_nanabanana.py` — Generate images using Nanobanana

**Utility:**
- `01_image_generation/04_rescue_batch.py` — Retry failed batches

**What they do:**
- Accept model configuration
- Generate paired images (neutral + with disability keyword)
- Save to `data_raw/{model}_pipeline/`
- Handle API rate limiting and retries

**How to run individually:**
```bash
python 01_image_generation/01_generate_flux.py
```

**How to run via pipeline:**
```bash
python run_pipeline.py --stages image_generation --models flux
```

---

### Stage 2: Attribute Extraction

**Primary Scripts:**
- `02_attribute_extraction/01_flux_attribute.py` — Extract FLUX image attributes
- `02_attribute_extraction/02_sd35_attribute.py` — Extract SD3.5 image attributes
- `02_attribute_extraction/03_nanobanana_attribute.py` — Extract Nanobanana attributes

**Alternate Versions (v2):**
- `02_attribute_extraction/01_flux_v2_attribute.py` — Alternate FLUX extraction
- `02_attribute_extraction/02_sd35_v2_attribute.py` — Alternate SD3.5 extraction

**Specialty:**
- `02_attribute_extraction/09_profession_erasure_batch.py` — Erasure bias experiment

**What they do:**
- Load all images from `data_raw/{model}_pipeline/`
- Call Vision Language Model (gpt-4-mini) for each image
- Extract 12 demographic attributes
- Save to `data_processed/{model}_attributes/attributes.csv`
- Handle VLM API rate limiting and retries

**How to run individually:**
```bash
python 02_attribute_extraction/01_flux_attribute.py
```

**How to run via pipeline:**
```bash
python run_pipeline.py --stages attribute_extraction --models flux,sd35
```

---

### Stage 3: Compliance Cleanup

**Active Script:**
- `00_purge_flux_teens.py` — Remove teen data (age 13-17)

**What it does:**
- Filters `attributes.csv` files to remove rows with age_bin = "13-17"
- Applied to all models before analysis
- Required for GDPR/legal compliance

**Triggered automatically by pipeline** in stage 3, but can be run manually:
```bash
python 00_purge_flux_teens.py
```

---

### Stage 4: Human Validation

**Script:**
- Handled by `run_pipeline.py` automatically
- Uses `data_processed/human_overrides.csv` as input

**What happens:**
- Pipeline loads `human_overrides.csv`
- For each correction, replaces VLM prediction with human-validated value
- Updates attributes CSVs in-memory before analysis

---

### Stage 5: Analysis & Chart Generation

**Primary Scripts (Run These):**
- `03_analysis/05_flux_analysis.py` — FLUX statistical analysis + charts
- `03_analysis/06_sd35_analysis.py` — SD3.5 statistical analysis + charts
- `03_analysis/03_nanobanana_analysis.py` — Nanobanana statistical analysis + charts

**What they do:**
- Load cleaned `attributes.csv`
- Compare "neutral" vs "conditioned" images
- Perform chi-squared tests for each demographic attribute
- Generate PNG charts comparing distributions
- Output individual CSV summaries
- Generate multi-page master Excel file

**Outputs:**
- `../results/charts/{model}_*.png` — Distribution comparison charts
- `../results/statistical_summary/{model}_statistical_summary.csv` — Test results
- `../results/DCDD_Final_Master_Analysis.xlsx` — Master report (aggregated)

**How to run individually:**
```bash
python 03_analysis/05_flux_analysis.py
```

**How to run via pipeline:**
```bash
python run_pipeline.py --stages analysis
```

---

### Utility & Specialty Scripts

**Analysis Utilities:**
- `03_analysis/08_human_audit.py` — Review human validation decisions
- `03_analysis/10_erasure_analysis.py` — Analyze "erasure" bias pattern
- `03_analysis/11_intersectional_analysis.py` — Analyze intersectional demographics
- `03_analysis/13_explicit_analysis_no_ambiguous.py` — Analysis excluding ambiguous predictions
- `03_analysis/03_nanobanana_charts.py` — Chart generation helper

**Validation Utilities:**
- `03_analysis/human_validation_of_nanobanana_ambiguous_img.py` — Validate Nanobanana predictions
- `03_analysis/human_validation_of_sd_flux_non_amb.py` — Validate FLUX/SD3.5 predictions

**Special Analysis:**
- `04_VLM_alignment_Flux/` — FLUX-specific VLM accuracy testing
- `mitigation/` — Bias mitigation experiments
- `plotting/` — Reusable visualization functions

---

## Running Scripts Manually

### Run a single model's pipeline:
```bash
python run_pipeline.py --models flux
```

### Run only analysis (skip generation/extraction):
```bash
python run_pipeline.py --stages analysis
```

### Run only a specific analysis script:
```bash
python 03_analysis/05_flux_analysis.py
```

### Validate data before analysis:
```bash
python validate_stage_outputs.py
```

### Dry run (see what would happen):
```bash
python run_pipeline.py --dry-run
```

---

## Configuration

All scripts read from `pipeline_config.yaml` (via `run_pipeline.py`).

To customize:
1. Edit `pipeline_config.yaml`
2. Run `python run_pipeline.py`

You can also override specific parameters in the config:
- VLM model: `analysis.vlm_model`
- Output DPI: `analysis.dpi`
- Models to process: `models` list
- Stages to run: `stages` list

---

## Logging & Debugging

Logs are saved to `logs/pipeline_run_YYYY-MM-DD_HH-MM-SS.log`

Check logs for:
- What stages ran and when
- Any errors or warnings
- Validation results
- Stage timing

View logs:
```bash
tail -f logs/pipeline_run_*.log
```

---

## Adding New Scripts

To add a new script to the pipeline:

1. **Create script** in appropriate directory:
   - Image generation: `01_image_generation/`
   - Attribute extraction: `02_attribute_extraction/`
   - Analysis: `03_analysis/`

2. **Update `pipeline_config.yaml`**:
   ```yaml
   models:
     - name: "my_model"
       gen_script: "01_image_generation/XX_my_model.py"
       extract_script: "02_attribute_extraction/XX_my_model.py"
       analysis_script: "03_analysis/XX_my_model.py"
   ```

3. **Run pipeline**:
   ```bash
   python run_pipeline.py --models my_model
   ```

---

## Script Maintenance

**Active scripts** are used by the pipeline regularly.

**Archive scripts** (in `03_analysis/archive/`) are kept for:
- Historical reference
- Reverting to old methodology
- Comparing alternate approaches

**Deprecated scripts** should be moved to `archive/` with a note explaining why.

---

## Common Tasks

### Re-run just the charts:
```bash
python run_pipeline.py --stages analysis --models flux,sd35,nanobanana
```

### Check if data is ready for analysis:
```bash
python validate_stage_outputs.py
```

### Use alternate data version:
1. Edit `pipeline_config.yaml`: change `data_folder` from `01_flux_attributes` to `01_flux_attributes_v2`
2. Run: `python run_pipeline.py --stages analysis --models flux`

### Manually correct a VLM mistake:
1. Add row to `data_processed/human_overrides.csv`
2. Re-run: `python run_pipeline.py --stages human_validation analysis`

---

## Next Steps

1. **Start here**: `python run_pipeline.py --help`
2. **Validate data**: `python validate_stage_outputs.py`
3. **Run full pipeline**: `python run_pipeline.py`
4. **Check results**: Open `../results/DCDD_Final_Master_Analysis.xlsx`

For detailed info, see:
- `../README.md` — Project overview
- `../PIPELINE.md` — Detailed execution guide
- `../data_processed/README.md` — Data format reference
