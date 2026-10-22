# Disability-Conditioned Demographic Drift (DCDD)

Official code and reproduction package for the paper:

> **Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models** (WACV 2027)
> Dataset: [`Red-Pathfinder/dcdd-wacv-2027`](https://huggingface.co/datasets/Red-Pathfinder/dcdd-wacv-2027)

---

## Overview

This repository reproduces all empirical results, statistical tests, tables, and figures reported in the paper.

The benchmark evaluates three contemporary text-to-image models—**FLUX.1 [dev]**, **Stable Diffusion 3.5 Large**, and **Nano Banana 2 (Google Imagen 3 via Gemini API)**—across 77 occupational prompts paired with disability conditioning (Mobility, Vision, Hearing, and Mental Health conditions). It measures:
1. **Representational drift**: Systematic shifts in perceived demographic attributes (gender, skin tone) and contextual dimensions (affective sentiment, physical framing, activity engagement, social integration).
2. **Algorithmic erasure**: Severe concentration of content-safety refusals on disability prompts (specifically mental health).
3. **Measurement reliability**: Formal human inter-annotator agreement and alignment between vision-language model predictions and human consensus.

---

## Quick Start

Requires Python 3.10+. Execution takes ~2 seconds on a standard laptop.

```bash
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run paper reproduction
bash reproduce.sh
```

The repository includes a local cache of the tabular benchmark files (`Drift/dataset_cache/`), so the pipeline runs immediately without network requests. If the cache is absent or if `--download` is passed, the script retrieves the pinned dataset revision (`cf80c45`) from Hugging Face.

### Optional: Full Image Dataset

To download the complete 1.6 GB dataset (2,472 generated images + 300 blinded reliability images):

```bash
bash reproduce.sh --mode full
```

*(Note: If the Hugging Face repository is set to private, export your access token first: `export HF_TOKEN="<your_token>"`)*.

---

## What `reproduce.sh` Produces

Running `bash reproduce.sh` verifies all SHA-256 data checksums and outputs:

| Manuscript Item | Description | Output Location |
|:---|:---|:---|
| **Table 2** | Human inter-annotator reliability (Fleiss' $\kappa$, nominal Krippendorff's $\alpha$, pairwise %, unanimous %) across 3 raters on 300 blinded stimulus images. | Terminal & Report |
| **Table 3** | Agreement % and Cohen's $\kappa$ between VLM classifications and human consensus ($N=300$). | Terminal & Report |
| **Table 4 & Fig. 1** | Safety filter refusal counts and condition rates ($145 / 385 = 37.7\%$ on mental health). | Terminal, Report & `charts/fig1_gemini_refusal_erasure.png` |
| **Tables 5, 7, 8, 9** | Pairwise Counterfactual Flip Rates (CFR %) across demographic, affective, and social agency dimensions. | Terminal & Report |
| **Tables 10, 11, 12 & Fig. 3** | Per-condition distribution shifts (bias-corrected Cramér's V, $\chi^2$ tests, FDR $q$-values, and directional shifts). | Terminal, Report & `charts/fig3_flux_silent_demographic_drift.png` |
| **Audit Report** | Comprehensive side-by-side audit markdown report comparing calculated metrics directly to published paper values. | `Drift/results/reproduction/DCDD_Paper_Reproduction_Report.md` |

---

## Dataset Structure & Counts

| Model | Planned Tasks | Manifest Attempts | Completed Images | Safety Refusals | Timeouts |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FLUX.1 [dev]** | 385 | 380 | 380 | 0 | 5 (unrecorded network timeouts) |
| **SD 3.5 Large** | 385 | 385 | 385 | 0 | 0 |
| **Nano Banana 2** | 1,925 | 1,925 | 1,707 | 218 | 0 |
| **Total** | **2,695** | **2,690** | **2,472** | **218** | **5** |

The released metadata files (`dataset_manifest.csv`, `annotations.csv`, `generation_metadata.csv`) catalog **2,690 recorded attempts** ($380 + 385 + 1,925$). Five initial FLUX batch requests timed out on the provider API without returning an image; accounting for these 5 timeouts reconciles the cataloged dataset to the 2,695 planned tasks ($2,690 + 5 = 2,695$).

---

## Citation Placeholder (Review Stage)

```bibtex
@inproceedings{dcdd2027,
  title={Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models},
  author={[Author details omitted for double-blind review]},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```
