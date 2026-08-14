# 01 — Execution Trace

> **Forensic Audit of the DCDD Repository**
> All statements are supported by source code. Nothing is inferred.

---

## 1. Repository Root

```
c:\Study\Coding\Research\DCDD\Drift\
```

---

## 2. Top-Level Orchestrator

**File:** `scripts/run_pipeline.py`
**Purpose:** CLI entry point that dispatches stages in sequence. It does **not** import any analysis
module directly; it calls them by subprocess conceptually but in the current code every stage
function body ends with `logger.info("[OK] … complete")` — i.e., the actual subprocess calls are
placeholder stubs. No `subprocess.run(analysis_script)` call exists in the file.

**Stages dispatched (in order):**
1. `image_generation` → per-model generation scripts (stubs in orchestrator)
2. `attribute_extraction` → per-model extraction scripts (stubs)
3. `compliance_cleanup` → `00_purge_flux_teens.py` (named in PIPELINE.md L131)
4. `human_validation` → reads `data_processed/human_overrides.csv`
5. `analysis` → per-model analysis scripts (stubs)

Because the orchestrator stages are stubs, the analysis scripts must be run **directly** to produce
results. The pipeline framework documents intent but does not execute the statistics scripts.

---

## 3. Scripts That Produce Final Statistical Outputs

The following scripts are confirmed to write statistical output files (CSV / XLSX):

| # | Script | Output File | Role |
|---|--------|-------------|------|
| A | `scripts/03_analysis/05_flux_analysis.py` | `results/statistical summary/flux_statistical_summary.csv` + PNG charts | **Primary** FLUX statistical analysis |
| B | `scripts/03_analysis/06_sd35_analysis.py` | `results/statistical summary/sd35_statistical_summary.csv` + PNG charts | **Primary** SD 3.5 statistical analysis |
| C | `scripts/03_analysis/03_nanobanana_charts.py` | `results/statistical summary/nanobanana_statistical_summary.csv` + PNG chart | **Primary** Nanobanana (Gemini) chart+stats script |
| D | `scripts/03_analysis/03_nanobanana_analysis.py` | `results/DCDD_Final_Analysis_with_bias_severity.xlsx` | Tri-model comparison Excel (9 attribute sheets) |
| E | `scripts/03_analysis/13_explicit_analysis_no_ambiguous.py` | `results/DCDD_Explicit_Analysis_No_Ambiguous.xlsx` | Ambiguity-removed re-analysis |
| F | `scripts/03_analysis/11_intersectional_analysis.py` | `results/DCDD_Intersectional_Bias_Results.xlsx` | Race×Gender intersectional test |
| G | `scripts/03_analysis/10_erasure_analysis.py` | `results/DCDD_Executive_Summary_Profession_Erasure.xlsx` | Profession-erasure descriptive table |

**Not included in the final pipeline (non-statistical, UI-only, or human annotation tools):**
- `scripts/03_analysis/08_human_audit.py` — interactive human annotation tool for Nanobanana ambiguous labels; writes to `human_overrides.csv`; not a statistical analysis
- `scripts/03_analysis/human_validation_of_sd_flux_non_amb.py` — interactive annotation tool for FLUX/SD3.5; writes to `human_overrides.csv`; not a statistical analysis
- `scripts/03_analysis/human_validation_of_nanobanana_ambiguous_img.py` — viewer-only (no write calls that save corrections); not a statistical analysis
- `scripts/03_analysis/archive/analysis_v2.py` — reads from `01_flux_attributes_v2/attributes_v2.csv` and `02_sd35_attributes_v2/attributes_v2.csv`, **both of which do not exist on disk**; will fail at `pd.read_csv()` and abort. This is an abandoned script.

---

## 4. Logical Execution Order

```
STAGE 0 — Compliance cleanup (pre-processing)
  scripts/00_purge_flux_teens.py
    → Filters age_bin == 'teen (13-17)' from:
      data_processed/01_flux_attributes/attributes.csv
      data_raw/flux_pipeline/metadata.csv
      JSONL batch files
    → Moves teen images to archive_teens/

STAGE 1 — Per-model primary analyses
  scripts/03_analysis/05_flux_analysis.py           [FLUX primary]
  scripts/03_analysis/06_sd35_analysis.py           [SD3.5 primary]
  scripts/03_analysis/03_nanobanana_charts.py       [Nanobanana/Gemini primary]

STAGE 2 — Tri-model cross-analysis (Excel output)
  scripts/03_analysis/03_nanobanana_analysis.py     [FLUX+SD35+Gemini combined]
  scripts/03_analysis/13_explicit_analysis_no_ambiguous.py  [Same, ambig removed]

STAGE 3 — Supplementary analyses
  scripts/03_analysis/11_intersectional_analysis.py [Race x Gender]
  scripts/03_analysis/10_erasure_analysis.py        [Profession erasure]
```

**No shared module or import chain exists.** Each script imports only standard libraries
(`pandas`, `numpy`, `scipy.stats`, `statsmodels`). There are no custom utility modules.

---

## 5. Data Input → Script Mapping (Confirmed by Code)

| Script | Input CSV (hard-coded path) | File Exists? |
|--------|----------------------------|--------------|
| `05_flux_analysis.py` L37 | `data_processed/01_flux_attributes/attributes.csv` | ✅ Yes |
| `06_sd35_analysis.py` L36 | `data_processed/02_sd35_attributes/attributes.csv` | ✅ Yes |
| `03_nanobanana_charts.py` L21 | `data_processed/03_nanobanana_attributes/nanobanana_attributes_final.csv` | ✅ Yes |
| `03_nanobanana_analysis.py` L9–11 | `01_flux_attributes_v2/attributes_v2.csv`, `02_sd35_attributes_v2/attributes_v2.csv`, `nanobanana_attributes_final.csv` | ❌ First two MISSING; third exists |
| `13_explicit_analysis_no_ambiguous.py` L9–11 | Same three paths as above | ❌ First two MISSING; third exists |
| `11_intersectional_analysis.py` L8–9 | `01_flux_attributes_v2/attributes_v2.csv`, `02_sd35_attributes_v2/attributes_v2.csv` | ❌ Both MISSING |
| `10_erasure_analysis.py` L6 | `results/profession_erasure.csv` | Cannot verify (results/ outside repo) |

**Critical finding:** Scripts D, E, and F reference `attributes_v2.csv` files that do not exist.
These scripts would fail at the `pd.read_csv()` call. The actual filenames on disk are
`flux_attributes_v2.csv` and `sd_attributes_v2.csv` (different naming). The `try/except` block
in D and E would catch the error and `return` immediately without producing output.

---

## 6. scripts/run_pipeline.py — Orchestrator Behaviour

The orchestrator reads `pipeline_config.yaml` which points the analysis stage to:
- `03_analysis/05_flux_analysis.py` (FLUX)
- `03_analysis/06_sd35_analysis.py` (SD3.5)
- `03_analysis/03_nanobanana_analysis.py` (Nanobanana)

However, `run_stage_analysis()` in `run_pipeline.py` only logs `"[OK] {model} analysis complete"`
without actually executing those scripts. The function body (L338–344) uses a try-block whose
only action is `logger.info("[OK]…")`. There is **no `subprocess.run()` or `importlib` call**.
