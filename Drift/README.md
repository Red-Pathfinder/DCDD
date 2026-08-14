# DCDD: Disability and Demographic Drift in Image Generation

## Overview

This project investigates whether state-of-the-art text-to-image generation models exhibit **demographic drift** when disability context is added to prompts. In other words: when you add the word "disability" or disability-related keywords to an image generation prompt, do the demographics of generated people change significantly?

**Research Question:**
> When disability keywords are added to image generation prompts, does the AI systematically shift demographic attributes (gender, race, age, etc.) compared to neutral baseline images?

### Key Finding

This research analyzes three major image generation models:
- **FLUX** (recent state-of-the-art)
- **Stable Diffusion 3.5** (popular open-source)
- **Nanobanana** (alternative model)

We measure whether demographic shifts occur in 12 attributes:
gender presentation, perceived race, skin tone, perceived age, sentiment, environment type, professional attire, gaze direction, physical framing, activity engagement, social integration, and image style.

---

## Quick Start

### One-Command Pipeline

Run the entire pipeline from image generation through analysis:

```bash
cd Drift
python scripts/run_pipeline.py
```

This will:
1. Generate images (if not already done)
2. Extract demographic attributes using Vision Language Model
3. Apply compliance filters (remove teen data)
4. Apply human validation corrections
5. Run statistical analysis and generate charts

### Check Validation

Before running analysis, validate that the pipeline state is correct:

```bash
python scripts/validate_stage_outputs.py
```

### Run Specific Stages

```bash
# Re-run only analysis
python scripts/run_pipeline.py --stages analysis

# Re-run only FLUX
python scripts/run_pipeline.py --models flux

# Run extraction for specific models
python scripts/run_pipeline.py --stages attribute_extraction --models sd35,nanobanana
```

---

## Project Structure

```
Drift/
├── pipeline_config.yaml           ← Central configuration (edit this to customize)
├── README.md                       ← This file
├── PIPELINE.md                     ← Detailed execution guide
│
├── data_raw/                       ← Generated images (raw output)
│   ├── flux_pipeline/              (3000+ images)
│   ├── sd35_pipeline/
│   └── nanobanana_2_pipeline/
│
├── data_processed/                 ← Extracted attributes & validation
│   ├── 01_flux_attributes/         (attributes.csv)
│   ├── 02_sd35_attributes/
│   ├── 03_nanobanana_attributes/
│   ├── human_overrides.csv         (manual corrections)
│   ├── vlm_alignment_audit.csv     (validation audit trail)
│   └── README.md                   (data format reference)
│
├── scripts/                        ← Pipeline automation scripts
│   ├── run_pipeline.py             ★ Main orchestrator (run this)
│   ├── validate_stage_outputs.py   ★ Validation before analysis
│   │
│   ├── 01_image_generation/        (generate images)
│   │   ├── 01_generate_flux.py
│   │   ├── 02_generate_sd35.py
│   │   └── 03_nanabanana.py
│   │
│   ├── 02_attribute_extraction/    (VLM extraction)
│   │   ├── 01_flux_attribute.py
│   │   ├── 02_sd35_attribute.py
│   │   └── 03_nanobanana_attribute.py
│   │
│   ├── 03_analysis/                (analysis & charts)
│   │   ├── 05_flux_analysis.py
│   │   ├── 06_sd35_analysis.py
│   │   ├── 03_nanobanana_analysis.py
│   │   └── archive/                (old scripts)
│   │
│   ├── 04_VLM_alignment_Flux/      (special analyses)
│   ├── mitigation/                 (bias mitigation experiments)
│   ├── plotting/                   (visualization utilities)
│   └── README.md                   (script organization)
│
└── ../results/                     ← Output files
    ├── charts/
    │   ├── flux charts/
    │   ├── sd35 charts/
    │   └── nanobanana charts/
    ├── statistical_summary/
    │   ├── flux_statistical_summary.csv
    │   ├── sd35_statistical_summary.csv
    │   └── nanobanana_statistical_summary.csv
    └── DCDD_Final_Master_Analysis.xlsx  ★ Master report
```

---

## Pipeline Overview

### Stage 1: Image Generation
Generate images with and without disability keywords using three models.
- Input: None (API calls)
- Output: 3000+ images per model
- Time: ~2-4 hours per model

### Stage 2: Attribute Extraction
Use Vision Language Model to extract demographic attributes from all generated images.
- Input: Images
- Output: `attributes.csv` (12 demographic columns)
- Time: ~1-2 hours per model

### Stage 3: Compliance Cleanup
Apply legal/ethical filters (remove teen data per GDPR/compliance requirements).
- Input: `attributes.csv`
- Output: Filtered `attributes.csv`
- Time: ~5 minutes

### Stage 4: Human Validation
Apply manual corrections from `human_overrides.csv` where VLM made mistakes.
- Input: `attributes.csv`, `human_overrides.csv`
- Output: Corrected `attributes.csv`
- Time: ~5 minutes

### Stage 5: Analysis & Reporting
Statistical analysis (chi-squared tests) and chart generation.
- Input: Corrected `attributes.csv`
- Output: Charts, statistics, master Excel report
- Time: ~10-20 minutes per model

---

## Key Files

### Configuration
- **`pipeline_config.yaml`** — Central configuration for all models and stages

### Scripts (Run These)
- **`scripts/run_pipeline.py`** — Main orchestrator (start here!)
- **`scripts/validate_stage_outputs.py`** — Pre-analysis validation

### Data
- **`data_processed/{model}_attributes/attributes.csv`** — Extracted attributes
- **`data_processed/human_overrides.csv`** — Manual corrections to VLM predictions
- **`data_processed/vlm_alignment_audit.csv`** — Validation audit trail

### Outputs
- **`../results/DCDD_Final_Master_Analysis.xlsx`** — Master report with all findings
- **`../results/charts/{model}_*.png`** — Individual demographic shift charts
- **`../results/statistical_summary/{model}_*.csv`** — Statistical test results

### Documentation
- **`PIPELINE.md`** — Detailed execution guide for each stage
- **`data_processed/README.md`** — Data format and schema reference
- **`scripts/README.md`** — Script organization and descriptions
- **`scripts/03_analysis/archive/README.md`** — Deprecated scripts

---

## Configuration: pipeline_config.yaml

The main configuration file controls:
- Which models to process
- Which pipeline stages to run
- Analysis parameters (DPI, colors, fonts)
- VLM model choice
- Compliance filters

### Example: Add a New Model

Edit `pipeline_config.yaml`:

```yaml
models:
  - name: "my_model"
    gen_script: "01_image_generation/04_my_model.py"
    extract_script: "02_attribute_extraction/04_my_model_extract.py"
    analysis_script: "03_analysis/04_my_model_analysis.py"
    data_folder: "04_my_model_attributes"
    output_prefix: "my_model"
```

Then run:
```bash
python scripts/run_pipeline.py --models my_model
```

### Example: Disable a Stage

Edit `pipeline_config.yaml`:

```yaml
stages:
  - id: 1
    name: "image_generation"
    enabled: false  # Skip image generation
```

---

## Data Schema

### `attributes.csv` Format

Each extracted attributes file contains:

| Column | Type | Description |
|--------|------|-------------|
| image_id | str | Unique image identifier |
| image_type | str | "neutral" or "conditioned" (disability keyword added) |
| gender_presentation | str | male, female, non-binary, ambiguous |
| perceived_race | str | Racial/ethnic classification |
| skin_tone | str | Skin tone classification |
| perceived_age | str | child, teen, young adult, adult, older adult |
| sentiment | str | positive, neutral, negative |
| environment_type | str | indoor, outdoor, professional, home, etc. |
| professional_attire | str | yes, no, partial, ambiguous |
| gaze_direction | str | toward camera, away, side, unclear |
| physical_framing | str | full body, upper body, head only |
| activity_engagement | str | passive, moderate, active |
| social_integration | str | alone, with others |
| image_style | str | artistic style or filter applied |

### `human_overrides.csv` Format

Manual corrections to VLM predictions:

| Column | Type | Description |
|--------|------|-------------|
| image_id | str | Image to correct |
| attribute | str | Which attribute to override |
| original_prediction | str | What VLM predicted |
| override_value | str | Correct value |
| reason | str | Why this correction was made |
| validator_name | str | Human who validated |
| validation_date | str | When validated |

---

## Compliance & Ethics

### Teen Data Removal

To comply with GDPR and protect minors, all data from people aged 13-17 is removed in Stage 3 before analysis.

This is **mandatory** and checked automatically:
```bash
python scripts/validate_stage_outputs.py
```

### Human Validation

The VLM (Vision Language Model) makes mistakes. We maintain a `human_overrides.csv` file where humans can correct misclassifications before analysis.

---

## Troubleshooting

### Issue: "Image count mismatch"
Re-run attribute extraction:
```bash
python scripts/run_pipeline.py --stages attribute_extraction
```

### Issue: Pipeline too slow
1. Check if API rate limits are throttling VLM calls
2. Run only analysis (skip generation/extraction):
   ```bash
   python scripts/run_pipeline.py --stages analysis
   ```

### Issue: Teen data still present
Run compliance cleanup again:
```bash
python scripts/run_pipeline.py --stages compliance_cleanup
```

### Issue: Can't find `pipeline_config.yaml`
Ensure you're in the `Drift/` directory:
```bash
cd Drift
python scripts/run_pipeline.py
```

---

## Output: Master Analysis File

After a complete run, open:
```
../results/DCDD_Final_Master_Analysis.xlsx
```

This Excel file contains:
- **Summary sheet**: Executive summary of findings
- **Flux results**: Demographic shifts for FLUX model
- **SD3.5 results**: Demographic shifts for Stable Diffusion 3.5
- **Nanobanana results**: Demographic shifts for Nanobanana
- **Statistical summary**: Chi-squared test results and p-values
- **Charts**: Visual comparison of demographic distributions

### Interpreting Results

- **Red cells**: Large demographic shift (high bias)
- **Amber cells**: Moderate demographic shift
- **Pale yellow cells**: Small demographic shift
- **p-value < 0.05**: Statistically significant (unlikely due to chance)

---

## Next Steps

1. **Run validator:** `python scripts/validate_stage_outputs.py`
2. **Review execution guide:** Read `PIPELINE.md`
3. **Run full pipeline:** `python scripts/run_pipeline.py`
4. **View results:** Check `../results/DCDD_Final_Master_Analysis.xlsx`

For detailed information:
- **How to run specific stages:** See `PIPELINE.md`
- **Data format details:** See `data_processed/README.md`
- **Script descriptions:** See `scripts/README.md`

---

## Citation & Contact

This research is part of the EMNLP paper on disability representation in AI-generated images.

**Corresponding Author**: [Your Name]

---

## License

This project data and analysis are provided for research purposes.

---

**Last Updated**: 2026-06-12
