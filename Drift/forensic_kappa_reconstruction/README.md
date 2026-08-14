# Forensic Kappa Reconstruction Audit

This directory contains the independent forensic audit of the human validation statistics (Cohen's κ) for the DCDD Drift repository.

## Files
- `validation_summary.md`: Explains the critical methodological findings and why Cohen's kappa cannot be mathematically reconstructed from the repository artifacts.
- `artifact_trace.md`: Detailed trace of the validation scripts, datasets, and pipeline logic.
- `cohen_kappa_results.csv`: Expected outputs showing N/A values and specific explanatory notes for each requested model/attribute combination.
- `scripts/`: Contains the forensic audit script used to analyze the dataset structures and attempt the kappa calculations.

## Core Finding
The human validation pipeline utilized a **selective sampling methodology**, only presenting images to human validators when the VLM label contained the keywords "ambiguous", "unclear", or "indeterminate". Confident predictions were completely ignored. 

Because Cohen's kappa requires two independent raters to evaluate the *same set of items*, it is mathematically impossible to reconstruct the dataset-wide statistic without inappropriately inferring that the human validator would have agreed with the VLM on all images they never reviewed. Thus, per the forensic rules, the reconstruction is halted.
