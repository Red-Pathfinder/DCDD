# Disability-Conditioned Demographic Drift (DCDD)

Official code and reproduction package for the research paper:

> **Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models** (Under review)<br>
> Dataset: [`Red-Pathfinder/dcdd-wacv-2027`](https://huggingface.co/datasets/Red-Pathfinder/dcdd-wacv-2027).

---

## Overview

This repository provides code and reproduction instructions for evaluating how text-to-image models (FLUX.1 [dev], Stable Diffusion 3.5 Large, and NanoBanana 2 / Gemini 3.1 Flash) shift demographic and contextual attributes when disability conditioning is added to neutral occupational prompts.

Code belongs in this GitHub repository; dataset assets (images, metadata manifests, VLM annotations, human corrections, and blinded reliability study records) are hosted on Hugging Face.

---

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the Hugging Face dataset repository is private, configure your read access:

```bash
export HF_TOKEN="<your_huggingface_read_token>"
# or: hf auth login
```

---

## Reproduction

Run the single reproduction script from the repository root:

```bash
# 1. Statistical reproduction (default: downloads ~1.2 MB metadata, runs in seconds)
bash reproduce.sh

# 2. Full dataset reproduction (downloads all 2,472 images + 300 blinded study images, ~1.6 GB)
bash reproduce.sh --mode full
```

### What this reproduces:
- **Inter-annotator reliability:** Fleiss' $\kappa$ across 3 independent raters on 300 blinded stimulus images (gender: 0.8205, activity: 0.8147, social integration: 0.8142, sentiment: 0.6995, skin tone: 0.6427, physical framing: 0.5761).
- **Demographic drift & flip rates:** Bias-corrected Cramér's V, $\chi^2$ independence tests, FDR-adjusted $q$-values, and Conditional Flip Rates (CFR %).
- **Algorithmic erasure:** Google Gemini safety filter refusal distribution across conditions (145/218 refusals concentrated on mental health prompts).
- **Publication figures:** Generated to `Drift/results/reproduction/charts/`.

*Note:* Statistical reproduction operates deterministically on the released dataset and does not require image-generation API keys. Synthesizing new images from scratch via third-party APIs (fal.ai, Gemini) is a separate historical pipeline located in `Drift/scripts/`.

---

## Dataset Structure & Counts

| Model | Planned Tasks | Manifest Attempts | Completed Images | Safety Refusals | Timeouts |
|:---|:---:|:---:|:---:|:---:|:---:|
| FLUX.1 [dev] | 385 | 380 | 380 | 0 | 5 (unrecorded fal.ai timeouts) |
| SD 3.5 Large | 385 | 385 | 385 | 0 | 0 |
| NanoBanana 2 | 1,925 | 1,925 | 1,707 | 218 | 0 |
| **Total** | **2,695** | **2,690** | **2,472** | **218** | **5** |

The released dataset manifests (`dataset_manifest.csv`, `annotations.csv`, `generation_metadata.csv`) catalog **2,690 recorded attempts** ($380 + 385 + 1,925$). The 5 missing prompts were unrecorded HTTP network timeouts during initial FLUX batch generation on fal.ai. Adding these 5 timeouts reconciles the cataloged dataset to the 2,695 planned experimental tasks ($2,690 + 5 = 2,695$).

---

## Citation Placeholder (Review Stage)

```bibtex
@article{dcdd2027,
  title={Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models},
  author={[Author details omitted for double-blind review]},
  note={Under review},
  year={2027}
}
```
