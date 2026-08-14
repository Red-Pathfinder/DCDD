# DCDD Pipeline Map

This document illustrates the complete execution pipeline for the DCDD reproducibility package.

## 1. Generation
**Scripts:** `01_generate_flux.py`, `02_generate_sd35.py`, `03_nanabanana.py`
**Models:** FLUX.1 Dev, SD 3.5 Large, Nano Banana 2
**Outputs:** Generated images, metadata CSVs
**Paper Section:** 3.2

↓

## 2. Annotation
**Scripts:** `01_flux_v2_attribute.py`, `02_sd35_v2_attribute.py`, `03_nanobanana_attribute.py`
**Model:** GPT-5 mini
**Outputs:** Raw visual attribute classifications
**Paper Section:** 3.3

↓

## 3. Human Validation
**Scripts:** `human_audit.py`, `human_validation_of_sd_flux_non_amb.py`
**Outputs:** `human_validation.csv` (human corrections applied to ambiguous GPT-5 mini outputs)
**Paper Section:** 3.5

↓

## 4. Statistical Analysis
**Scripts:** `05_flux_analysis.py`, `06_sd35_analysis.py`, `03_nanobanana_analysis.py`, etc.
**Outputs:** Chi-square results, Cramér's V effect sizes, FDR corrections
**Paper Section:** 4

↓

## 5. Figures
**Scripts:** `calc_flip_rate.py`, `mental_health_collapse.py`, `plot_flux_gender_inversion.py`
**Outputs:** Publication figures (Counterfactual Flip Rates, Mental Health Collapse, Gender Inversion)
**Paper Section:** 4 & 5
