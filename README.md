# Disability-Conditioned Demographic Drift (DCDD)

Official code and reproduction package for the research paper:

> **Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models**
> *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2027)*
> Dataset: [`Red-Pathfinder/dcdd-wacv-2027`](https://huggingface.co/datasets/Red-Pathfinder/dcdd-wacv-2027)

---

## Abstract & Scope

This repository provides the canonical reproduction pipeline for the empirical findings, statistical evaluations, and figures reported in the paper.

The benchmark evaluates three contemporary text-to-image architectures—**FLUX.1 [dev]** (12B rectified flow transformer), **Stable Diffusion 3.5 Large** (8B MMDiT), and **Nano Banana 2** (Google Imagen 3 via Gemini API)—across 77 occupational prompts paired counterfactually with disability conditioning (Mobility, Vision, Hearing, and Mental Health conditions). The analysis examines:
1. **Representational Drift**: Systematic shifts in perceived demographic attributes (gender presentation, skin tone) and contextual framing (affective sentiment, physical framing, activity engagement, social integration).
2. **Algorithmic Erasure**: Disproportionate concentration of content-safety policy filter refusals on disability prompts (specifically mental health).
3. **Measurement Reliability**: Inter-annotator agreement across independent human raters and empirical alignment between vision-language model predictions and human consensus.

---

## Environment Setup

The reproduction suite is operating system-independent (tested on Linux, macOS, and Windows) and requires Python 3.10 or higher. Simply copy and paste the block corresponding to your operating system:

### Windows (PowerShell / Command Prompt)

```powershell
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
python reproduce.py
```

### Linux (bash)

```bash
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

*(POSIX users may alternatively invoke `bash reproduce.sh`).*

### macOS (zsh / bash)

```bash
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

### Option B: Conda Environment (Cross-Platform)

```bash
git clone https://github.com/Red-Pathfinder/DCDD.git
cd DCDD
conda env create -f environment.yml
conda activate dcdd
python reproduce.py
```

---

## Reproduction Protocol

The statistical pipeline operates directly on the released benchmark data and executes on standard CPU hardware without requiring GPU acceleration or external API credentials.

### Primary Execution

From the repository root, run the Python reproduction entry point:

```bash
python reproduce.py
```

Upon first execution, the pipeline automatically retrieves the required tabular benchmark files from the pinned Hugging Face repository (`Red-Pathfinder/dcdd-wacv-2027` @ revision `4ec3dd0`) into `Drift/dataset_cache/` and cryptographically verifies their SHA-256 checksums against `checksums.sha256`. Subsequent runs execute entirely offline from this verified local cache.

Upon completion, the pipeline displays aligned terminal tables and automatically opens the standalone academic reproduction report (`DCDD_Paper_Reproduction_Report.html`) in your default web browser, displaying camera-ready booktabs tables and embedded publication figures (`fig1_gemini_refusal_erasure.png` and `fig3_flux_silent_demographic_drift.png`). To suppress automatic browser opening in headless or CI environments, pass `--no-open`:

```bash
python reproduce.py --no-open
```

### Full Dataset Verification (Images)

To download and verify the complete 1.6 GB asset collection (2,472 generated stimulus images and 300 blinded human reliability study images):

```bash
python reproduce.py --mode full
```

*(If accessing a private Hugging Face repository, authenticate prior to execution: `export HF_TOKEN="<your_token>"`).*

---

## Reproducibility Guardrails & Long-Term Provenance

To guarantee that this analysis remains reproducible over extended multi-year periods, the repository incorporates the following structural guardrails:

1. **Automated Integrity Verification & Offline Caching**: Ingested benchmark files are validated against immutable SHA-256 signatures upon retrieval and cached in `Drift/dataset_cache/`, allowing subsequent evaluations to execute completely offline.
2. **Cryptographic Integrity Verification**: Every benchmark file is hashed upon ingestion; execution will halt if local data diverges by a single bit from `checksums.sha256`.
3. **Immutable Remote Storage**: The underlying image repository is pinned to an immutable commit hash on Hugging Face (`4ec3dd0e906c04e059a4073c18a69cb75597eba3`).
4. **Environment Specification**: Dependencies are specified with backwards-compatible version bounds in `requirements.txt` and mirrored in `environment.yml` for Conda.
5. **Runtime Validation**: Python version ($\ge 3.10$) and all required scientific libraries are checked before execution, providing explicit diagnostic guidance if dependencies are missing.

---

## Manuscript Items Produced

Upon completion, the reproduction script displays aligned terminal tables and exports artifacts to `Drift/results/reproduction/`:

| Manuscript Item | Scientific Description | Output Destination |
|:---|:---|:---|
| **Table 2** | Human inter-annotator reliability metrics (Fleiss' $\kappa$, nominal Krippendorff's $\alpha$, pairwise agreement %, unanimous %; $N=300$ images, 3 independent raters). | Terminal, Report, & `tables/table2_inter_annotator_reliability.csv` |
| **Table 3** | Vision-language model alignment metrics (percentage agreement and Cohen's $\kappa$ against majority human consensus; $N=300$). | Terminal, Report, & `tables/table3_vlm_human_agreement.csv` |
| **Table 4 & Fig. 1** | Safety filter refusal counts and condition rates ($145 / 385 = 37.7\%$ refusal rate for mental health prompts). | Terminal, Report, `tables/table4_gemini_refusal_rates.csv`, & `charts/fig1_gemini_refusal_erasure.png` |
| **Tables 5, 7, 8, 9** | Pairwise Counterfactual Flip Rates (CFR %) across demographic, affective, and social agency dimensions. | Terminal, Report, & `tables/table{5,7,8,9}_*.csv` |
| **Tables 10, 11, 12 & Fig. 3** | Per-condition distribution shifts (bias-corrected Cramér's V, $\chi^2$ independence tests, FDR $q$-values, and directional shift identification). | Terminal, Report, `tables/table{10,11,12}_*.csv`, & `charts/fig3_flux_silent_demographic_drift.png` |
| **Visual Report (HTML)** | Standalone academic reproduction report with LaTeX booktabs styling, horizontally scrollable tables, and embedded publication figures. | `Drift/results/reproduction/DCDD_Paper_Reproduction_Report.html` |
| **Consolidated Audit** | Comprehensive side-by-side markdown comparison contrasting every calculated metric against reported manuscript figures. | `Drift/results/reproduction/DCDD_Paper_Reproduction_Report.md` |

---

## Upstream Generation & Annotation Pipeline

For researchers looking to re-generate stimulus images from scratch, extract visual attributes with new Vision-Language Models, or audit newly released text-to-image models using the DCDD benchmark protocol, complete upstream pipelines are documented in [`Supplementary Material/`](Supplementary%20Material/):

- **Image Generation (`Supplementary Material/01_generation/`)**:
  - `01_generate_flux.py`: Generates FLUX.1 [dev] stimulus batches via fal.ai API.
  - `02_generate_sd35.py`: Generates Stable Diffusion 3.5 Large batches via fal.ai API.
  - `03_nanabanana.py`: Generates Nano Banana 2 batches across temperature sweeps ($0.0$–$1.0$) via Google Gemini API.
- **VLM Attribute Extraction (`Supplementary Material/02_annotation/`)**:
  - `01_flux_v2_attribute.py`, `02_sd35_v2_attribute.py`, `03_nanobanana_attribute.py`: Extracts the 12 visual dimensions from generated stimulus images.
  - `Annotation_Codebook.docx` & `Annotation_Protocol.docx`: Formal codebook defining all 12 attribute definitions, permissible values, and ambiguous-label resolution rules.
- **Human Verification & Measurement Reliability (`Supplementary Material/03_validation/`)**:
  - `human_audit.py`, `vlm_alignment_flux.py`: Evaluates inter-rater agreement and VLM-to-human consensus alignment across independent annotators.

---

## Benchmark Structure & Accounting

| Model Architecture | Planned Tasks | Manifest Attempts | Completed Images | Safety Refusals | Timeouts |
|:---|:---:|:---:|:---:|:---:|:---:|
| **FLUX.1 [dev]** | 385 | 380 | 380 | 0 | 5 (unrecorded network timeouts) |
| **Stable Diffusion 3.5 Large** | 385 | 385 | 385 | 0 | 0 |
| **Nano Banana 2** | 1,925 | 1,925 | 1,707 | 218 | 0 |
| **Total** | **2,695** | **2,690** | **2,472** | **218** | **5** |

The released metadata files catalog **2,690 recorded generation attempts** ($380 + 385 + 1,925$). Five initial FLUX batch requests encountered HTTP gateway timeouts on the generation infrastructure without returning an image; accounting for these 5 network timeouts reconciles the cataloged dataset to the 2,695 planned experimental tasks ($2,690 + 5 = 2,695$).

---

## Citation

```bibtex
@inproceedings{dcdd2027,
  title={Disability-Conditioned Demographic Drift: Intersectional Stereotyping and Algorithmic Erasure in Text-to-Image Generative Models},
  author={[Author details omitted for double-blind review]},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```
