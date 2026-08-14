# 07 — Manuscript Claims Verification

> **Forensic Audit of the DCDD Repository**
> Each claim is evaluated against repository code only. No inference. No assumption.
> Evidence is cited by file and line number.

---

## Evaluation Key

| Verdict | Meaning |
|---------|---------|
| ✅ SUPPORTED | Code directly confirms the claim |
| ⚠️ PARTIALLY SUPPORTED | Code partially confirms; important caveats exist |
| ❌ NOT SUPPORTED | Code does not confirm; or directly contradicts |
| 🔍 CANNOT VERIFY | Claim depends on data/conditions not determinable from code alone |

---

## Claim 1: "We compute Cramér's V."

**Verdict: ✅ SUPPORTED — with caveat**

**Evidence:**
- `05_flux_analysis.py` lines 115–124: `cramers_v()` function defined and called
- `06_sd35_analysis.py` lines 114–124: identical function defined and called
- `03_nanobanana_charts.py` lines 57–65: identical function defined and called
- `03_nanobanana_analysis.py` lines 32–34: `calculate_cramers_v()` called at line 122
- `13_explicit_analysis_no_ambiguous.py` lines 35–37: same
- `11_intersectional_analysis.py` lines 21–27: same

**Caveat:** Two different formulas are used. Scripts 05, 06, and 03_charts use a
**bias-corrected** formula (Bergsma/Szekely). Scripts 03_nanobanana_analysis, 11, and 13 use
the **classic uncorrected** formula. The numerical results differ. The threshold scales also
differ (V < 0.10 = Negligible in 05/06 vs V < 0.05 = Negligible in 03_nanobanana_analysis).

---

## Claim 2: "We use Benjamini-Hochberg FDR."

**Verdict: ⚠️ PARTIALLY SUPPORTED**

**Evidence:**
- `05_flux_analysis.py` line 164: `multipletests(..., alpha=0.05, method="fdr_bh")` ✅
- `06_sd35_analysis.py` line 163: same ✅
- `03_nanobanana_charts.py` line 94: same ✅
- `03_nanobanana_analysis.py` line 144: same ✅ (but script is currently inoperative)
- `13_explicit_analysis_no_ambiguous.py` line 154: same ✅ (but inoperative)

**Caveats:**
1. `11_intersectional_analysis.py` does **not** apply BH — uses raw p < 0.05 (line 116)
2. `10_erasure_analysis.py` does **not** perform any statistical test
3. BH scope is inconsistent: 12 tests global per model in 05/06; 9 in 03_charts; per-attribute
   families in 03_nanobanana_analysis and 13_explicit
4. The term "global BH" (across all models and attributes combined) does not appear anywhere

---

## Claim 3: "We evaluate 9 dimensions."

**Verdict: ⚠️ PARTIALLY SUPPORTED**

**Evidence:**
- `03_nanobanana_charts.py` line 30–34: `VISUAL_ATTRIBUTES` list has **9** entries ✅
- `03_nanobanana_analysis.py` line 17–21: `VISUAL_ATTRIBUTE_COLS` has **9** entries ✅
- `13_explicit_analysis_no_ambiguous.py` line 20–24: **9** entries ✅

**Caveats:**
- `05_flux_analysis.py` and `06_sd35_analysis.py` each test **12** attributes, not 9:
  ```
  VISUAL_ATTRIBUTES = [
      "gender_presentation", "perceived_race", "skin_tone", "perceived_age",
      "sentiment", "environment_type", "professional_attire", "gaze_direction",
      "physical_framing", "activity_engagement", "social_integration", "image_style"
  ]
  ```
- The number of dimensions tested differs by model and script. The claim "9 dimensions" matches
  the Nanobanana-model analysis scripts and the v2-schema attributes list in `pipeline_config.yaml`
  (which also has 12 but matches 9 for the v2 data schema).
- If the claim refers to the Nanobanana analysis: SUPPORTED.
- If it refers to all models: NOT SUPPORTED (12 for FLUX and SD3.5).

---

## Claim 4: "We compute Chi-square tests."

**Verdict: ✅ SUPPORTED**

**Evidence:**
- Every statistical analysis script calls `scipy.stats.chi2_contingency()`:
  - `05_flux_analysis.py` lines 138, 141 (in both branches of `test_attribute()`)
  - `06_sd35_analysis.py` lines 137, 140 (identical)
  - `03_nanobanana_charts.py` line 80
  - `03_nanobanana_analysis.py` line 119
  - `13_explicit_analysis_no_ambiguous.py` line 129
  - `11_intersectional_analysis.py` line 107

**Caveat:** In scripts 05 and 06, when sparse expected counts are detected (`n_sparse > 0`),
the code labels the test `"Fisher/Merged Fallback"` but still executes `chi2_contingency()`.
The Fisher exact test is **never actually called**. The label is misleading.

---

## Claim 5: "We report adjusted q-values."

**Verdict: ❌ NOT SUPPORTED**

**Evidence:**
- The output column is named `"FDR-Corrected P-Value"` in scripts 05 and 06 (line 191 in both)
- The values stored are BH-adjusted p-values returned by `multipletests()` as second return value
- Storey's q-value estimation is not imported or called anywhere in the repository
- The term "q-value" does not appear anywhere in the codebase

**What is reported:** BH-adjusted p-values (not Storey q-values). If the manuscript uses
"q-values" to mean "FDR-corrected p-values (BH method)", then the claim is ⚠️ PARTIALLY SUPPORTED
as a terminological issue, not a methodological one.

---

## Claim 6: "We remove Safety Refusal images."

**Verdict: ⚠️ PARTIALLY SUPPORTED — varies by script**

**Evidence:**
- In `05_flux_analysis.py` and `06_sd35_analysis.py`: Safety Refusals are **not present** in the
  FLUX and SD3.5 attribute CSVs (confirmed by data inspection). Removal is not needed and not done.
- In `03_nanobanana_charts.py`: Safety Refusals are **included** in the chi-square test as a
  category. Comment at line 80 of `03_nanobanana_analysis.py` explicitly states: `"We must include
  'Safety Refusal' in our categories if it exists"`.
- In `13_explicit_analysis_no_ambiguous.py` (lines 18, 55–59): Safety Refusal IS excluded via
  `EXCLUSION_KEYWORDS = ['ambiguous', 'unclear', 'indeterminate', 'safety refusal']`
- In `08_human_audit.py` line 67: `if str(row.get('image_style')) == 'Safety Refusal': continue`
  (skipped during human annotation)
- In `human_validation_of_nanobanana_ambiguous_img.py` line 38: same skip logic

**Summary:** Safety Refusals are excluded in `13_explicit_analysis_no_ambiguous.py` and in the
human annotation tools. They are **included** in `03_nanobanana_charts.py` and
`03_nanobanana_analysis.py`. They are absent from FLUX/SD3.5 data entirely.

---

## Claim 7: "We exclude ambiguous labels."

**Verdict: ⚠️ PARTIALLY SUPPORTED — depends on script and metric**

**Evidence:**
- **For CFR metric** in 05 and 06: Ambiguous labels are explicitly excluded (lines 106–107):
  `if n_val != 'ambiguous' and c_val != 'ambiguous'`
- **For chi-square tests** in 05, 06, 03_nanobanana_charts, 03_nanobanana_analysis: Ambiguous
  labels are **NOT excluded** — they appear as columns in the contingency table
- **In 13_explicit_analysis_no_ambiguous.py**: Ambiguous labels ARE excluded (EXCLUSION_KEYWORDS
  at line 18, applied via `is_explicit()` filter)
- **Ambiguous counts in FLUX v1 data:** `gender_presentation` = 20 rows; `perceived_race` = 96 rows

**Summary:** The claim is true for CFR and for the explicit-only analysis script. It is false
for the primary chi-square tests in the main analysis scripts.

---

## Claim 8: "We compute effect sizes."

**Verdict: ✅ SUPPORTED**

**Evidence:**
- Cramér's V computed in all statistical scripts (see Claim 1)
- CFR (Counterfactual Flip Rate) computed in `05_flux_analysis.py` and `06_sd35_analysis.py` as
  an additional effect measure
- Profession erasure drop computed in `10_erasure_analysis.py`
- Intersectional proportion shift computed in `11_intersectional_analysis.py`

---

## Claim 9: "We normalize for unequal sample sizes."

**Verdict: ❌ NOT SUPPORTED**

**Evidence:**
- The chi-square test itself accounts for sample size via expected frequencies
- No explicit normalization (e.g., proportion weighting, IPTW, balanced sampling) is applied
  before feeding data to chi-square
- In `compute_proportions()` (used for visualization only): proportions are computed as
  `100 * count / total`, but this function is used only for bar chart rendering, not for
  statistical tests
- The bias-corrected Cramér's V formula adjusts for table dimensions but does not balance
  unequal group sizes (75 neutral vs 305 conditioned in FLUX)
- No documentation or code evidence of any normalization procedure applied to the statistical tests

---

## Additional Findings Not Covered by the Listed Claims

### Finding A: Data Path Failures in Three Scripts

`03_nanobanana_analysis.py`, `13_explicit_analysis_no_ambiguous.py`, and
`11_intersectional_analysis.py` all reference CSV paths that do not exist:
- Expected: `01_flux_attributes_v2/attributes_v2.csv`
- Actual on disk: `01_flux_attributes_v2/flux_attributes_v2.csv`

These three scripts will exit at the try/except block without producing any statistical output.

### Finding B: "Fisher/Merged Fallback" Label Is Misleading

Scripts 05 and 06 write `"Fisher/Merged Fallback"` to the "Test Used" column when expected
cell counts < 5. However, no Fisher exact test is called. Both branches of the conditional
execute `chi2_contingency(tab, correction=False)`. The label in the CSV output does not reflect
the actual test performed.

### Finding C: Inconsistent BH Scope

- Scripts 05 and 06: BH on 12 tests, global per model
- Script 03_nanobanana_charts: BH on 9 tests, global per model
- Scripts 03_nanobanana_analysis and 13_explicit: BH on ~4–12 tests per attribute

A truly global correction (across all models and all attributes simultaneously) is not performed
anywhere.

### Finding D: Yates' Continuity Correction

Scripts 05, 06, and 03_nanobanana_charts explicitly disable Yates' correction:
`chi2_contingency(tab, correction=False)`. Scripts 03_nanobanana_analysis, 13_explicit, and 11
do not set the `correction` parameter, defaulting to `True` for 2×2 tables and `False` for
larger tables.

### Finding E: Pooled vs. Per-Condition Design

The primary analysis scripts (05, 06, 03_charts) pool all disability conditions into a single
"conditioned" group for the chi-square test. This tests whether the overall distribution of the
conditioned group differs from neutral. Per-condition tests (testing each disability separately)
are only implemented in `03_nanobanana_analysis.py` and variants — which are currently inoperative.

### Finding F: Human Overrides Not Applied in Analysis Scripts

The `human_overrides.csv` file (1,703 corrections across 850 Nanobanana and 306 FLUX image IDs)
is never read by any analysis script (05, 06, 03_charts, 03_analysis, 11, 13). The overrides may
or may not be pre-baked into the attribute CSVs — this cannot be determined from code alone.

### Finding G: Teen Filter Is a No-Op

The comment in `05_flux_analysis.py` (line 77) states: "POST-HOC TRUNCATION: Slices the FLUX
dataset from 505 to 385 images." However, the FLUX attributes.csv already contains 380 rows
with no teen entries. The filter removes 0 rows at runtime. The SD3.5 attributes.csv similarly
contains 385 rows with no teen entries.

### Finding H: Inconsistent Cramér's V Thresholds

Two different threshold scales are used:
- Implementation 1 (05, 06, 03_charts): Negligible < 0.10 ≤ Small < 0.20 ≤ Medium < 0.40 ≤ Large
- Implementation 2 (03_analysis, 11, 13): Negligible < 0.05 ≤ Small < 0.15 ≤ Moderate < 0.30 ≤ Large

A given V value can receive different labels depending on which script computed it.
