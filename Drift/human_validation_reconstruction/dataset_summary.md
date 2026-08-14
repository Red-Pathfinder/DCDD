# Final Dataset Summary

This document identifies the final datasets within the repository that correspond to the unified six-attribute evaluation schema used in the manuscript.

## Identified Final Datasets

### 1. FLUX.1 Dev
* **Final CSV**: `data_processed/01_flux_attributes_v2/flux_attributes_v2.csv`
* **Image Folder**: `data_raw/flux_pipeline/`
* **Total Images**: 381

### 2. Stable Diffusion 3.5 Large
* **Final CSV**: `data_processed/02_sd35_attributes_v2/sd_attributes_v2.csv`
* **Image Folder**: `data_raw/sd35_pipeline/`
* **Total Images**: 386

### 3. Nano Banana (Gemini 3.1 Flash)
* **Final CSV**: `data_processed/03_nanobanana_attributes/nanobanana_attributes_final.csv`
* **Image Folder**: `data_raw/nanobanana_2_pipeline/`
* **Total Images**: 1,926

## Retained Attributes
These datasets contain the final unified taxonomy. The six evaluated attributes are:
1. `gender_presentation`
2. `skin_tone`
3. `sentiment`
4. `physical_framing`
5. `activity_engagement`
6. `social_integration`

*(Note: Older schemas/columns such as `perceived_race`, `image_style`, and `gaze_direction` are present in the datasets but will be strictly ignored by the validation pipeline, per the manuscript's methodology).*
