# 04 — Cramér's V Implementation

> **Forensic Audit of the DCDD Repository**
> Two distinct implementations of Cramér's V coexist in this repository.

---

## Overview

The repository contains **two different formulas** for Cramér's V, used in different scripts:

| Implementation | Formula Type | Scripts |
|---------------|-------------|---------|
| **Bias-corrected** (Bergsma 2013 / Szekely-Rizzo) | Subtracts bias term from φ² | `05_flux_analysis.py`, `06_sd35_analysis.py`, `03_nanobanana_charts.py` |
| **Classic uncorrected** | Standard textbook formula | `03_nanobanana_analysis.py`, `11_intersectional_analysis.py`, `13_explicit_analysis_no_ambiguous.py` |

---

## Implementation 1: Bias-Corrected Cramér's V

### Files

- `scripts/03_analysis/05_flux_analysis.py`, lines 115–124
- `scripts/03_analysis/06_sd35_analysis.py`, lines 114–124 (identical)
- `scripts/03_analysis/03_nanobanana_charts.py`, lines 57–65 (identical)

### Exact Code

```python
def cramers_v(contingency_table: np.ndarray) -> float:
    """Calculates Cramér's V (Effect Size)."""
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    phi2 = max(0, (chi2 / n) - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
```

### Mathematical Formula

**Step 1 — Bias-corrected φ²:**

$$\tilde{\phi}^2 = \max\!\left(0,\; \frac{\chi^2}{n} - \frac{(r-1)(c-1)}{n-1}\right)$$

**Step 2 — Corrected dimensions:**

$$\tilde{r} = r - \frac{(r-1)^2}{n-1}, \qquad \tilde{c} = c - \frac{(c-1)^2}{n-1}$$

**Step 3 — Cramér's V:**

$$\tilde{V} = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r}-1,\; \tilde{c}-1)}}$$

### Properties

- **Bias correction:** YES — the formula is the bias-corrected version attributed to
  Bergsma (2013) / Szekely & Rizzo. It subtracts `(r-1)(c-1)/(n-1)` from φ² before computing V.
- **Floor at zero:** YES — `max(0, ...)` prevents negative φ² values
- **Denominator guard:** `if denom > 0 else 0.0` prevents division by zero

### Parameters supplied

- `contingency_table`: the 2 × K numpy array produced by `crosstab().values.astype(float)`
- Internal chi2 call: `correction=False` (Yates' continuity correction disabled)
- `n = contingency_table.sum()` = total observation count
- `r, c = contingency_table.shape` = number of rows (2) and columns (K categories)

### Thresholds

These effect-size labels are applied **only to label the output CSV**; they are also displayed
on chart annotations. They are hard-coded in `label_effect()` inside `run_statistical_pipeline()`:

```python
def label_effect(v: float) -> str:
    if v < 0.10: return "Negligible"
    if v < 0.20: return "Small"
    if v < 0.40: return "Medium"
    return "Large"
```

| Label | Cramér's V range |
|-------|-----------------|
| Negligible | V < 0.10 |
| Small | 0.10 ≤ V < 0.20 |
| Medium | 0.20 ≤ V < 0.40 |
| Large | V ≥ 0.40 |

These thresholds are **hard-coded** and are used in:
1. The exported statistical summary CSV (column "Effect Size")
2. The chart annotation text (e.g., `"V=0.35 | CFR=12.3%"`)

---

## Implementation 2: Classic (Uncorrected) Cramér's V

### Files

**Script 03_nanobanana_analysis.py**, lines 32–34:
```python
def calculate_cramers_v(chi2, n, min_dim):
    if n == 0 or min_dim <= 1: return 0.0
    return np.sqrt(chi2 / (n * (min_dim - 1)))
```

**Script 13_explicit_analysis_no_ambiguous.py**, lines 35–37 (identical):
```python
def calculate_cramers_v(chi2, n, min_dim):
    if n == 0 or min_dim <= 1: return 0.0
    return np.sqrt(chi2 / (n * (min_dim - 1)))
```

**Script 11_intersectional_analysis.py**, lines 21–27:
```python
def calculate_cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n == 0 or min(r, k) < 2: return 0.0
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))
```

### Mathematical Formula

$$V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, c) - 1)}}$$

This is the **original Cramér (1946) formula**, without bias correction.

### Properties

- **Bias correction:** NO
- **Denominator:** `n * (min_dim - 1)` where `min_dim = min(r, c)`
- In `03_nanobanana_analysis.py` and `13_explicit_analysis_no_ambiguous.py`:
  `min_dim = min(obs_clean.shape)` passed as argument
- In `11_intersectional_analysis.py`:
  `min_dim = min(r, k)` computed from `.shape` of the contingency matrix

### Thresholds

Same labels as Implementation 1, used in `get_bias_severity()`:

```python
def get_bias_severity(v_stat):
    if v >= 0.30: return "Large Bias 🔴"
    if v >= 0.15: return "Moderate Bias 🟠"
    if v >= 0.05: return "Small Bias 🟡"
    return "Negligible"
```

**Note:** These thresholds differ from Implementation 1 thresholds:

| Label | Impl 1 threshold | Impl 2 threshold |
|-------|-----------------|-----------------|
| Negligible | V < 0.10 | V < 0.05 |
| Small | 0.10 ≤ V < 0.20 | 0.05 ≤ V < 0.15 |
| Medium/Moderate | 0.20 ≤ V < 0.40 | 0.15 ≤ V < 0.30 |
| Large | V ≥ 0.40 | V ≥ 0.30 |

These thresholds are hard-coded and used in the Excel output column "Bias Severity".

---

## Numerical Difference Between Implementations

Verified empirically with a 2×3 test table (N=610):

| Metric | Value |
|--------|-------|
| Bias-corrected V (Impl 1) | 0.155729 |
| Uncorrected V (Impl 2) | 0.165819 |
| Absolute difference | 0.010089 |

The bias-corrected formula consistently returns **lower** values than the uncorrected formula for
the same contingency table. This creates a systematic inconsistency in effect size reporting across
scripts that use different implementations.

---

## Summary

- **Is Cramér's V computed?** YES — in all six statistical analysis scripts
- **Is bias correction applied?** YES in scripts 05, 06, 03_charts; NO in scripts 03_analysis, 11, 13
- **Denominator:** min(r-1, c-1) after bias correction in Impl 1; n×(min(r,c)-1) in Impl 2
- **Thresholds hard-coded?** YES in both implementations; different scales used
- **Thresholds used in visualization?** YES — appear on chart annotations and in Excel "Bias Severity" column
