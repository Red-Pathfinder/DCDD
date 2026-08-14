# Forensic Validation Summary & Cohen's Kappa Reconstruction Failure

## Objective
To independently reconstruct the Cohen's κ statistics reported in the manuscript using the available repository artifacts.

## Conclusion: Reconstruction Impossible
**Sufficient artifacts do not exist to reconstruct Cohen's κ. The calculation cannot be performed without improperly inferring missing data.**

Per the forensic rules: *"Before computing any statistics, determine whether sufficient artifacts exist to reconstruct Cohen's κ exactly. If not, explain what is missing and stop. Do not attempt to infer or approximate missing values."*

### Why Cohen's Kappa Cannot Be Computed
Cohen's kappa ($\kappa$) is a statistical measure of inter-rater reliability. It requires two independent raters (e.g., the VLM and a Human) to independently categorize the **same set of items**.

A forensic audit of `human_validation_of_sd_flux_non_amb.py` and `08_human_audit.py` reveals a fundamental methodological flaw in how human validation was conducted:

1. **Selective Validation Bias**: The human validator did not review the full dataset, nor did they review a representative random sample. The validation scripts filter the dataset using an `is_ambiguous(val)` function. **The human was only presented with images where the VLM explicitly output an "ambiguous", "unclear", or "indeterminate" label.** 
2. **Missing Human Labels**: For every image where the VLM was confident (e.g., predicted "White presenting"), the image bypassed human review entirely. Consequently, there is no independent "Human Label" for the vast majority of the dataset.
3. **The Inference Fallacy**: To compute a dataset-wide Cohen's kappa, one would have to assume that the human validator would have perfectly agreed with the VLM on all non-ambiguous images they never actually looked at. This is mathematically invalid and artificially inflates agreement to near 100%. If one only computes kappa on the subset of ambiguous images that *were* reviewed, the VLM's responses have zero variance (the VLM label is always "ambiguous"), which causes the expected agreement formula to break down, resulting in a mathematically meaningless kappa value (often 0 or undefined).

### Specific Attribute Failures
- **Sentiment (FLUX & SD 3.5)**: The script `human_validation_of_sd_flux_non_amb.py` contains explicit logic to skip sentiment validation entirely (`if attr_name == "sentiment" and val_lower == "neutral/unclear": return False`). The human validator literally never reviewed sentiment for these models. Any reported kappa for sentiment on FLUX or SD 3.5 is completely unsupported by the repository code.
- **VLM Alignment Audit**: The script `vlm_alignment_flux.py` did conduct a true independent audit on confident predictions, but it was restricted to a tiny subset: exactly 50 images, only on FLUX, and exclusively targeting the `vision impairment` condition. It cannot be used to generate dataset-wide or condition-agnostic kappa values. Furthermore, the resulting CSV (`vlm_alignment_audit.csv`) contains 22 exact duplicate rows due to a bug in the script.

## Manuscript Validation Discrepancies
Because the statistics cannot be reconstructed, any specific Cohen's $\kappa$ values (e.g., "$\kappa = 0.85$") or overall agreement percentages reported in the manuscript are **unsupported by the repository artifacts.** 

If the manuscript claims to have calculated Cohen's kappa across the entire dataset, that claim is incorrect based on the pipeline's selective sampling of ambiguous-only labels. 

If the manuscript reports sample sizes ($N$) equal to the full dataset size for inter-rater reliability, those sample sizes are incorrect because the human only rated the subset of images recorded in `human_overrides.csv`.
