# DCDD Scripts & Reproduction Pipeline

This directory contains the canonical paper reproduction engine and archival generation code for the DCDD benchmark.

## Canonical Reproduction (WACV 2027)

To reproduce all reported paper results from the published benchmark data:

```bash
# Recommended: execute from the repository root
python reproduce.py

# Or directly from this directory:
python reproduce_paper.py
```

### Analysis Performed by `reproduce_paper.py`

1. **Cryptographic Verification**: Verifies SHA-256 hashes for all 11 benchmark tabular files against `checksums.sha256`.
2. **Reliability Analysis (Tables 2 & 3)**:
   - Computes Fleiss' $\kappa$, nominal Krippendorff's $\alpha$, pairwise agreement %, and unanimous agreement % across 3 independent human raters on 300 blinded stimulus images.
   - Computes Cohen's $\kappa$ and alignment rates between VLM predictions and majority human consensus.
3. **Safety Refusal Analysis (Table 4 & Figure 1)**:
   - Evaluates Google Gemini content-filter refusals across all 5 prompt conditions, computing refusal counts and condition rates ($145 / 385 = 37.7\%$ on mental health prompts).
4. **Counterfactual Flip Rates (Tables 5, 7, 8, 9)**:
   - Computes pairwise flip rates (CFR %) across demographic, affective, and social agency dimensions for FLUX.1 [dev], Stable Diffusion 3.5 Large, and Nano Banana 2.
5. **Distributional Shifts (Tables 10, 11, 12 & Figure 3)**:
   - Computes $\chi^2$ independence tests, bias-corrected Cramér's V, and FDR $q$-value corrections, identifying directional stereotyping shifts.
6. **Report Generation**:
   - Saves high-resolution publication figures to `../results/reproduction/charts/`.
   - Generates a side-by-side audit report at `../results/reproduction/DCDD_Paper_Reproduction_Report.md`.

The statistical pipeline executes deterministically on standard CPU hardware without requiring GPU acceleration or third-party API dependencies.

---

## Archival Data Collection Scripts

The subdirectories in this folder preserve the original data collection code used during initial benchmark construction:
- `01_image_generation/`: API dispatch scripts calling commercial providers (fal.ai for FLUX.1 [dev] and Stable Diffusion 3.5 Large; Google GenAI for Gemini / Nano Banana 2).
- `02_attribute_extraction/`: Vision-language model harnesses used to extract visual attributes from generated images.
- `03_analysis/`: Model-specific exploratory analysis scripts used during early development.

> **Methodological Note:** These data collection scripts require commercial API credentials and local disk storage. They are retained for provenance and auditability, but are not required for reproducing any finding in the published paper. All reported results are computed directly from the released benchmark data via `reproduce_paper.py`.
