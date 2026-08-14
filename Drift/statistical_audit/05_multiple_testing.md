# 05 — Multiple Testing Correction

> **Forensic Audit of the DCDD Repository**
> Every statement is drawn directly from source code.

---

## Is Benjamini-Hochberg Actually Performed?

**YES** — in five of the seven analysis scripts. **NO** in `10_erasure_analysis.py` and
`11_intersectional_analysis.py`.

---

## Implementation Details

### Library Call (all scripts that use BH)

```python
from statsmodels.stats.multitest import multipletests
reject, p_corr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

The `multipletests` function (statsmodels) implements the Benjamini-Hochberg (1995) procedure.
The four return values are:
1. `reject` — boolean array: True if test is significant after correction
2. `p_corr` — array of BH-adjusted p-values
3. `_` — FWER-corrected p-values (alphacsidak/Bonferroni step-down) — discarded
4. `_` — corrected alpha values — discarded

**Alpha value:** `0.05` in all scripts.

---

## Script-by-Script Breakdown

---

### Script 05_flux_analysis.py

**Location:** `run_statistical_pipeline()`, line 164

```python
reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
results["fdr_corrected_p"] = p_corr
results["significant_fdr"] = reject
```

**Input:** `results["raw_p"]` — a pandas Series of raw p-values from `test_attribute()`, one per
attribute.

**Number of p-values in correction family:** **12**
(one per attribute in `VISUAL_ATTRIBUTES` list)

**Scope:** Single model (FLUX), all attributes pooled, applied once.

**Output columns:**
- `fdr_corrected_p` — BH-adjusted p-value
- `significant_fdr` — boolean (True if reject H₀ after BH at α=0.05)

**Are q-values reported separately from adjusted p-values?** NO — BH-adjusted p-values are stored
in the column named `"FDR-Corrected P-Value"` in the exported CSV. They are BH-adjusted p-values,
not q-values in the Storey (2002) sense.

**Are raw p-values also reported?** YES — column `"Raw P-Value"` in the exported CSV.

---

### Script 06_sd35_analysis.py

**Location:** `run_statistical_pipeline()`, line 163 (identical to 05)

```python
reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
```

**Number of p-values:** **12** (same 12 attributes)
**Scope:** Single model (SD 3.5), all attributes pooled, applied once.
**Output:** Same structure as FLUX.

---

### Script 03_nanobanana_charts.py

**Location:** `run_statistical_pipeline()`, line 94

```python
reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
```

**Number of p-values:** **9**
(9 attributes in `VISUAL_ATTRIBUTES`: `gender_presentation`, `perceived_race`, `skin_tone`,
`sentiment`, `gaze_direction`, `physical_framing`, `activity_engagement`, `social_integration`,
`image_style`)

**Scope:** Single model (Nanobanana/Gemini), all attributes pooled, applied once.

**Note:** This script does NOT include `perceived_age`, `environment_type`, `professional_attire`
in its attribute list (unlike 05 and 06 which have 12 attributes).

---

### Script 03_nanobanana_analysis.py

**Location:** `main()`, line 144

```python
reject, pvals_corrected, _, _ = multipletests(p_values_to_correct, alpha=0.05, method='fdr_bh')
```

**Structure:** The BH correction is called **once per attribute column** (inside the `for col in
VISUAL_ATTRIBUTE_COLS` loop). The `p_values_to_correct` list accumulates p-values from all
(model × condition) pairs tested for that attribute.

**Number of p-values per BH call:** Verified empirically:
- For the Nanobanana data alone (only Gemini model, 4 conditions): **4 p-values per attribute**
- If FLUX and SD3.5 v2 data were present (3 models × 4 conditions): would be **12 p-values per attribute**

**Current operative state:** Since FLUX v2 and SD3.5 v2 CSVs are missing, the try/except at
lines 56–69 would exit immediately. The script produces no output. The BH call is never reached.

**Scope if operative:** Per-attribute BH families — NOT global correction across all attributes.
Each of the 9 attributes has its own separate BH family.

---

### Script 13_explicit_analysis_no_ambiguous.py

**Location:** `main()`, line 154 (identical structure to 03_nanobanana_analysis.py)

```python
reject, pvals_corrected, _, _ = multipletests(p_values_to_correct, alpha=0.05, method='fdr_bh')
```

**Same BH scope as 03_nanobanana_analysis.py** (per attribute, not global).

**Current operative state:** Same path failure as 03_nanobanana_analysis.py.

---

### Script 11_intersectional_analysis.py

**NO BH correction.** Raw p-values are compared directly to 0.05:

```python
"Significant (p < 0.05)": "YES" if p_val < 0.05 else "NO"
```

Line 116. No `multipletests` import or call in this script.

---

### Script 10_erasure_analysis.py

**NO statistical test at all.** Purely descriptive arithmetic. No p-values produced.

---

## Comparison of BH Scope Across Scripts

| Script | # tests corrected together | Scope |
|--------|---------------------------|-------|
| 05_flux_analysis.py | 12 | All attributes, 1 model, global |
| 06_sd35_analysis.py | 12 | All attributes, 1 model, global |
| 03_nanobanana_charts.py | 9 | All attributes, 1 model, global |
| 03_nanobanana_analysis.py | ~4 per attribute | Per-attribute family |
| 13_explicit_analysis.py | ~4 per attribute | Per-attribute family |
| 11_intersectional_analysis.py | 0 (no correction) | None |
| 10_erasure_analysis.py | 0 (no statistics) | None |

**Critical inconsistency:** The BH scope is not uniform across scripts. Scripts 05 and 06 use a
global family of 12 tests. Scripts 03_nanobanana_analysis and 13_explicit apply per-attribute
families of 4 tests. Script 11 applies no correction at all.

---

## Are Adjusted P-values or Q-values Reported?

**Scripts 05 and 06:** BH-adjusted p-values are reported in the column named `"FDR-Corrected P-Value"`.
These are the adjusted p-values returned by `multipletests`, not Storey q-values.

**Scripts 03_nanobanana_analysis and 13_explicit:** BH-adjusted p-values stored as `"FDR p-value"`
column in each Excel sheet.

**Raw p-values:** Also retained in all cases.

**Q-values (Storey method):** Not computed anywhere.

---

## Summary

| Property | Value |
|----------|-------|
| BH performed | YES (in 5 of 7 scripts) |
| Alpha | 0.05 |
| Library | `statsmodels.stats.multitest.multipletests(..., method='fdr_bh')` |
| BH scope — FLUX | 12 tests (global per model) |
| BH scope — SD3.5 | 12 tests (global per model) |
| BH scope — Nanobanana charts | 9 tests (global per model) |
| BH scope — tri-model analyses | 4 tests per attribute (per attribute) |
| Correction global across models | NO |
| Q-values (Storey) | NOT computed |
| Bonferroni | NOT used |
