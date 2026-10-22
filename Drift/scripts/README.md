# DCDD Scripts & Reproduction Pipeline

This directory contains the canonical paper reproduction pipeline and archival generation code for the DCDD benchmark.

## Canonical Reproduction (WACV 2027)

To reproduce all reported paper results from the published benchmark data:

```bash
# Recommended: run from the repository root
bash reproduce.sh

# Or directly from this directory:
python scripts/reproduce_paper.py
```

### What `reproduce_paper.py` Does

1. **Integrity Verification**: Verifies SHA-256 checksums for all 11 tabular benchmark files against `checksums.sha256`.
2. **Reliability Analysis (Tables 2 & 3)**:
   - Computes Fleiss' $\kappa$, nominal Krippendorff's $\alpha$, pairwise agreement %, and unanimous agreement % across 3 independent human annotators ($N=300$ blinded images).
   - Computes Cohen's $\kappa$ and alignment rates between VLM predictions and majority human consensus.
3. **Safety Refusal Analysis (Table 4 & Figure 1)**:
   - Evaluates Google Gemini content-filter refusals across all 5 prompt conditions, computing refusal counts and condition rates ($145 / 385 = 37.7\%$ on mental health prompts).
4. **Counterfactual Flip Rates (Tables 5, 7, 8, 9)**:
   - Computes pairwise flip rates (CFR %) across demographic, affective, and social agency dimensions for FLUX.1 [dev], SD 3.5 Large, and Nano Banana 2.
5. **Distributional Shifts (Tables 10, 11, 12 & Figure 3)**:
   - Computes $\chi^2$ independence tests, bias-corrected Cramér's V, and FDR $q$-value corrections, recovering directional shifts (such as the female presentation shift in FLUX.1 [dev] under vision impairment).
6. **Report Generation**:
   - Saves high-resolution publication figures to `../results/reproduction/charts/`.
   - Generates a side-by-side verification report at `../results/reproduction/DCDD_Paper_Reproduction_Report.md`.

Execution completes in ~2 seconds on any standard laptop or workstation without requiring GPUs or external API credentials.

---

## Archival Data Collection Scripts

The subdirectories here contain the original data collection code used to construct the benchmark:
- `01_image_generation/`: Prompt dispatch scripts calling third-party APIs (fal.ai for FLUX.1 [dev] and SD 3.5 Large; Google GenAI for Gemini / Nano Banana 2).
- `02_attribute_extraction/`: Vision-language model prompt harnesses used to extract visual attributes from generated images.
- `03_analysis/`: Early exploratory and model-specific drafting scripts used during initial experimentation.

> **Note for Researchers:** These scripts require commercial API keys and local GPU / network storage. They are preserved for provenance and methodology audit, but are **not** needed to reproduce any finding in the paper. All published results are derived directly from the released tabular benchmark via `reproduce_paper.py`.
