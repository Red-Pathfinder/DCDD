# Data Processing Directory - Structure & Versioning

This directory contains extracted demographic attributes from generated images, processed outputs, and validation data.

## Directory Structure

```
data_processed/
├── 01_flux_attributes/          # FLUX model attribute extraction (primary version)
├── 01_flux_attributes_v2/       # FLUX attribute extraction (v2 - experimental/alternate)
├── 02_sd35_attributes/          # Stable Diffusion 3.5 attributes (primary)
├── 02_sd35_attributes_v2/       # SD3.5 attributes (v2 - experimental/alternate)
├── 03_nanobanana_attributes/    # Nanobanana model attributes (primary)
├── erasure batch/               # Attributes for erasure bias analysis
├── human_overrides.csv          # Manual corrections to VLM predictions
├── vlm_alignment_audit.csv      # Validation audit trail
└── CHANGELOG.md                 # This file
```

## File Formats

### `attributes.csv` Format

Each `attributes.csv` contains demographic attributes extracted by Vision Language Model (currently gpt-4-mini):

**Columns:**
- `image_id` — Unique image identifier
- `image_type` — "neutral" or "conditioned" (whether disability keyword was added)
- `gender_presentation` — Perceived gender (male, female, non-binary, ambiguous)
- `perceived_race` — Perceived racial/ethnic group
- `skin_tone` — Skin tone classification
- `perceived_age` — Age category (child, teen, young adult, adult, older adult, etc.)
- `sentiment` — Emotional expression (positive, neutral, negative)
- `environment_type` — Setting (indoor, outdoor, professional, home, etc.)
- `professional_attire` — Professional dress assessment
- `gaze_direction` — Direction of eye contact/gaze
- `physical_framing` — Body positioning and framing
- `activity_engagement` — Level of activity (passive, moderate, active)
- `social_integration` — Social context (alone, with others, etc.)
- `image_style` — Artistic style or filter

### `human_overrides.csv` Format

Manual corrections to VLM predictions:

**Columns:**
- `image_id` — Image identifier
- `attribute` — Which attribute to correct (e.g., "gender_presentation")
- `original_prediction` — VLM's prediction
- `override_value` — Corrected value
- `reason` — Why the correction was made (e.g., "ambiguous", "clear error")
- `validator_name` — Human who made the correction
- `validation_date` — When the correction was made

## Versioning & Alternates

### Why Multiple Versions?

Version multiplicity typically indicates one of the following:

1. **Methodology iteration**: v2 represents a different VLM prompt, extraction logic, or filtering approach
2. **Bug fix**: v2 corrects issues found in v1
3. **Experiment**: v2 is an alternate extraction method for comparison

### Current Pipeline Default

The **active pipeline** uses these as defaults:
- `01_flux_attributes/` for FLUX analysis
- `02_sd35_attributes/` for SD3.5 analysis
- `03_nanobanana_attributes/` for Nanobanana analysis

To use an alternate version, modify `pipeline_config.yaml`:

```yaml
models:
  - name: "flux"
    data_folder: "01_flux_attributes_v2"  # Use v2 instead
```

## CHANGELOG

### 2026-06-12

**v1 Status Documentation**
- Renamed `01_flux_attributes/` → implicit v1 (primary pipeline version)
- Documented `01_flux_attributes_v2/` existence without overwriting primary
- Created version decision documentation

### Reasons for Keeping Multiple Versions

- **Reproducibility**: Allows re-running analysis with different extraction methodologies
- **Audit trail**: Tracks evolution of attribute extraction accuracy
- **Experimentation**: Enables A/B comparison of VLM prompts/models

## Data Validation

Before analysis, validate that:

1. All attributes.csv files have the same columns
2. Image counts match between attributes.csv and raw images
3. No missing values in critical columns
4. image_type values are "neutral" or "conditioned"

Run the validation script:

```bash
python scripts/validate_stage_outputs.py
```

## Manual Overrides Application

Human overrides are applied after VLM extraction:

1. Load `attributes.csv`
2. Load `human_overrides.csv`
3. For each row in overrides, replace the predicted value with the override value
4. Save corrected CSV
5. Log which rows were modified

This happens automatically in stage 4 (human_validation) of the pipeline.

## Compliance Filtering

**Age restriction**: Teen data (age_bin = "13-17") is removed per legal/compliance requirements.

Location: Applied in stage 3 (compliance_cleanup)
Affected files: All `attributes.csv` files
Status: Required before analysis

## Adding New Data Versions

If you create a new extraction version:

1. Create folder: `04_your_model_attributes/`
2. Extract attributes and save to `attributes.csv`
3. Update `pipeline_config.yaml` to add the model
4. Validate with `validate_stage_outputs.py`
5. Update this CHANGELOG with the date and reason for the new version
