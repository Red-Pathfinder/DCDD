# 02 — Statistics Used

> **Forensic Audit of the DCDD Repository**
> Every entry below is documented with the exact file, function, and library call.

---

## Summary Table

| Method | Used? | Scripts |
|--------|-------|---------|
| Chi-square test (Pearson) | ✅ Yes | 05, 06, 03_charts, 03_analysis, 11, 13 |
| Fisher's exact test | ❌ No (label only — not actually called) | 05, 06, 03_charts |
| Cramér's V (bias-corrected) | ✅ Yes | 05, 06, 03_charts |
| Cramér's V (uncorrected) | ✅ Yes | 03_analysis, 11, 13 |
| Benjamini-Hochberg FDR | ✅ Yes | 05, 06, 03_charts, 03_analysis, 13 |
| Bonferroni correction | ❌ Not used anywhere |
| Odds ratios | ❌ Not used anywhere |
| Confidence intervals | ❌ Not used anywhere |
| Bootstrap | ❌ Not used anywhere |
| Permutation test | ❌ Not used anywhere |
| Counterfactual Flip Rate (CFR) | ✅ Yes (custom metric) | 05, 06 |
| Profession erasure rate | ✅ Yes (descriptive only) | 10 |
| Proportion shift (intersectional) | ✅ Yes (descriptive) | 11 |

---

## 1. Chi-Square Test

### 1a. Scripts 05_flux_analysis.py and 06_sd35_analysis.py

**Function:** `test_attribute(df, attribute)` — lines 127–150 in both scripts (identical code).

**Library call:**
```python
from scipy.stats import chi2_contingency
_, p, _, _ = chi2_contingency(tab, correction=False)
```

**Parameters:**
- `correction=False` — Yates' continuity correction is **disabled** in all cases
- Input `tab`: a 2 × K numpy float array (rows = [neutral, conditioned]; columns = unique label categories)

**What data are supplied:**
- The full dataset (`df`) after teen-bin filter, with all 380 rows (FLUX) or 385 rows (SD3.5)
- `tab` is built via `pd.crosstab(df['image_type'], df[attribute]).reindex(['neutral', 'conditioned']).dropna(how='all').values.astype(float)`
- ALL conditioned images across all 4 disability conditions are pooled into a single 'conditioned' group
- No other row filtering occurs before crosstab (ambiguous labels remain; Safety Refusals not present in FLUX/SD3.5 data)

**Sparse-cell detection (label only — not acting on it):**
```python
exp = expected_freq(tab)
n_sparse = (exp < 5).sum()
if n_sparse > 0:
    test_used = "Fisher/Merged Fallback"   # label set here ...
    _, p, _, _ = chi2_contingency(tab, correction=False)  # ... but chi2 called anyway
else:
    test_used = "Chi-Square"
    _, p, _, _ = chi2_contingency(tab, correction=False)
```
**Critical finding:** Both branches call `chi2_contingency(tab, correction=False)`. The label
`"Fisher/Merged Fallback"` is recorded in the output CSV column "Test Used" but **no Fisher exact
test is ever called**. There is no `fisher_exact()` call anywhere in these scripts.

---

### 1b. Script 03_nanobanana_charts.py

**Function:** `test_attribute(df, attribute)` — lines 67–85.

**Library call:** Same as above:
```python
_, p, _, _ = chi2_contingency(tab, correction=False)
```

**What data are supplied:** Full nanobanana dataset (1,925 rows); all 4 conditions pooled into
`conditioned`; CONDITION_COL = `image_type` derived by harmonizing `condition` column.

**Same Fisher label but chi2 execution issue as 05/06.**

---

### 1c. Script 03_nanobanana_analysis.py

**Function:** `main()` — lines 108–122.

**Library call:**
```python
chi2, p_val, dof, expected = chi2_contingency(obs_clean)
```

**Note:** `correction` parameter not specified → defaults to `True` for 2×2 tables (Yates' correction
enabled by scipy default), `False` for larger tables.

**What data are supplied:**
- `obs`: 2-row array [neutral_counts, cond_counts], where neutral = all neutral images for that model,
  cond = images for ONE specific disability condition for that model
- `obs_clean`: zero-columns removed via `obs[:, obs.sum(axis=0) > 0]`
- Safety Refusal rows are **included** as a category (explicit comment at line 80: `"We must include Safety Refusal"`)

---

### 1d. Script 13_explicit_analysis_no_ambiguous.py

**Function:** `main()` — lines 118–132.

**Library call:**
```python
chi2, p_val, dof, expected = chi2_contingency(obs_clean)
```

**What data are supplied:**
- Same structure as 1c, BUT rows where the tested attribute contains 'ambiguous', 'unclear',
  'indeterminate', or 'safety refusal' are **removed** before the test (via `is_explicit()` filter
  at line 87–88)
- Safety Refusal rows ARE excluded here (EXCLUSION_KEYWORDS at line 18 includes 'safety refusal')

---

### 1e. Script 11_intersectional_analysis.py

**Function:** `main()` — lines 101–110.

**Library call:**
```python
chi2, p_val, dof, expected = chi2_contingency(obs)
```

**What data are supplied:**
- `obs`: list of [neutral_count, cond_count] pairs for each Race×Gender intersectional category
- Input is FLUX v2 + SD3.5 v2 CSVs — **both paths are missing on disk** (`attributes_v2.csv`)
- If files were present: no ambiguous filter; no Safety Refusal filter; correction default

---

## 2. Cramér's V

See Phase 4 document (`04_cramers_v.md`) for full breakdown.

**Short summary:**
- **Bias-corrected variant** used in: `05_flux_analysis.py`, `06_sd35_analysis.py`, `03_nanobanana_charts.py`
- **Non-bias-corrected variant** used in: `03_nanobanana_analysis.py`, `11_intersectional_analysis.py`, `13_explicit_analysis_no_ambiguous.py`

---

## 3. Benjamini-Hochberg FDR Correction

**Library:** `from statsmodels.stats.multitest import multipletests`

### 3a. Scripts 05_flux_analysis.py and 06_sd35_analysis.py

**Function:** `run_statistical_pipeline(df)` — line 164 in both:
```python
reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
```
**Scope:** 12 p-values (one per attribute), applied once per model.
**Output fields:** `fdr_corrected_p`, `significant_fdr` (boolean).

### 3b. Script 03_nanobanana_charts.py

**Function:** `run_statistical_pipeline(df)` — line 94:
```python
reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
```
**Scope:** 9 p-values (one per attribute), applied once.

### 3c. Scripts 03_nanobanana_analysis.py and 13_explicit_analysis_no_ambiguous.py

**Function:** `main()` — line 144 (both scripts):
```python
reject, pvals_corrected, _, _ = multipletests(p_values_to_correct, alpha=0.05, method='fdr_bh')
```
**Scope:** Per attribute sheet — called once per attribute column, on the p-values from all
(model × condition) pairs for that attribute. Family size = 4 p-values (1 model × 4 conditions).

---

## 4. Counterfactual Flip Rate (CFR)

**Scripts:** `05_flux_analysis.py` and `06_sd35_analysis.py` (identical implementation).

**Function:** `calculate_flip_rate(df, attribute)` — lines 90–112 in both.

**Algorithm (custom — no library call):**
```python
for _, row in cond_df.iterrows():
    pair_id = row['pair_key']
    if pair_id in neutral_df.index:
        n_val = str(neutral_df.loc[[pair_id]].iloc[0][attribute]).lower().strip()
        c_val = str(row[attribute]).lower().strip()
        if n_val != 'ambiguous' and c_val != 'ambiguous':
            if n_val != c_val:
                flips += 1
            total_pairs += 1
return (flips / total_pairs * 100) if total_pairs > 0 else 0.0
```

**Key properties:**
- Ambiguous labels excluded from BOTH neutral and conditioned sides before counting
- Metric: percentage of matched pairs where attribute label changed
- pair_key = `seed + "_" + profession + "_" + age_bin`
- Safety Refusal not relevant (absent from FLUX/SD3.5 data)

---

## 5. Profession Erasure Rate

**Script:** `10_erasure_analysis.py`

**No statistical test.** Descriptive only:
```python
baseline_adherence = neutral_yes / len(neutral_df) * 100
cond_adherence = cond_yes / len(cond_df) * 100
erasure_drop = baseline_adherence - cond_adherence
```

No chi-square, no BH, no p-value.

---

## 6. Intersectional Proportion Shift

**Script:** `11_intersectional_analysis.py`

Additional descriptive metric alongside chi-square:
```python
neutral_pct = neutral_counts / neutral_counts.sum()
cond_pct = cond_counts / cond_counts.sum()
delta = cond_pct.subtract(neutral_pct, fill_value=0)
true_shift_cat = delta.idxmax()
shift_magnitude = delta.max() * 100
```

No BH correction applied. Significance reported as raw `p < 0.05`.
