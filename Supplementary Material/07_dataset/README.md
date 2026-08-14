# DCDD Benchmark Dataset

This folder contains the core benchmark dataset described in the WACV 2027 submission: *Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models*.

## File Inventory

| File | Description | Rows | Primary Key |
|------|-------------|------|-------------|
| `dataset_manifest.csv` | Master index of generated samples | 2690 | `sample_id` |
| `prompts.csv` | Unique text prompts | 1150 | `prompt_id` |
| `generation_metadata.csv` | Technical API parameters | 2690 | `sample_id` |
| `annotations.csv` | GPT-5 mini attribute classifications | 2690 | `sample_id` |
| `human_validation.csv` | Human corrections to ambiguous labels | 1703 | (`master_id`, `attribute_name`) |

## Data Dictionary

### dataset_manifest.csv
- **`sample_id`**: Globally unique identifier.
- **`model`**: Model used (`FLUX.1 Dev`, `SD 3.5 Large`, `Nano Banana 2`).
- **`pair_id`**: Groups neutral and conditioned images for counterfactual comparison.
- **`seed`**: Generation seed.
- **`sector`** & **`profession`**: Occupational grouping.
- **`age_bin`**: Age group.
- **`image_type`**: `neutral` or `conditioned`.
- **`disability`**: Condition applied (empty for neutral).

### prompts.csv
Contains the exact text strings sent to the models, deduplicated by parameters.

### generation_metadata.csv
Captures the generation parameters (Section 3.2).
- **`api_model_id`**: Exact API endpoint (e.g., `gemini-3.1-flash`).
- **`seed`**: Latent seed (FLUX, SD 3.5).
- **`temperature`**: Sampling temperature (Nano Banana 2).

### annotations.csv
Contains the GPT-5 mini visual classifications (Section 3.3).
The formal evaluation focuses on 6 core dimensions:
1. `Skin Tone`
2. `Gender Presentation`
3. `Affective Sentiment`
4. `Physical Framing`
5. `Activity Engagement`
6. `Social Integration`

*(Note: 12 attributes are provided in the file to ensure operational compatibility with the repository's statistical scripts).*

### human_validation.csv
Records human corrections applied to ambiguous VLM predictions (Section 3.5).
- **`master_id`**: Raw image identifier matching the validation scripts.
- **`attribute_name`**: The dimension corrected.
- **`corrected_value`**: The human-assigned label.

## Dataset Summary

The dataset spans 78 profession-age cells (26 professions × 3 age bins). Each cell contains one neutral baseline and four disability-conditioned variants.

| Model | Total Samples | Neutral | Conditioned |
|-------|--------------|---------|-------------|
| FLUX.1 Dev | 380 | 76 | 304 |
| SD 3.5 Large | 385 | 77 | 308 |
| Nano Banana 2 | 1925 | 385 | 1540 |
| **Total** | **2690** | **538** | **2152** |

*(Nano Banana 2 uses a 5-point temperature sampling sweep, resulting in a higher sample count).*
