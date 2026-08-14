# 06 — Hypothesis Test Count

> **Forensic Audit of the DCDD Repository**
> All counts are derived from executed code only, not from the manuscript.

---

## Counting Method

For each script, hypothesis count = the number of times `chi2_contingency()` is called with data
that would generate a p-value entered into statistical reporting.

---

## Script 05_flux_analysis.py

**Function:** `test_attribute()` called in `run_statistical_pipeline()` for each attribute.

**Call count:**
```
12 attributes × 1 chi2_contingency call per attribute × 1 model = 12 tests
```

**Attributes tested:**
1. `gender_presentation`
2. `perceived_race`
3. `skin_tone`
4. `perceived_age`
5. `sentiment`
6. `environment_type`
7. `professional_attire`
8. `gaze_direction`
9. `physical_framing`
10. `activity_engagement`
11. `social_integration`
12. `image_style`

**Design of comparison:**
- 1 pooled test: ALL conditioned images (across all 4 disability conditions) vs ALL neutral images
- NOT stratified by disability condition

**BH family size:** 12 p-values

**Total p-values for FLUX:** **12**

---

## Script 06_sd35_analysis.py

Structurally identical to 05_flux_analysis.py.

**Total p-values for SD 3.5:** **12**

---

## Script 03_nanobanana_charts.py

**Function:** `test_attribute()` called for each attribute.

**Call count:**
```
9 attributes × 1 chi2_contingency call per attribute × 1 model = 9 tests
```

**Attributes tested (9, NOT 12):**
1. `gender_presentation`
2. `perceived_race`
3. `skin_tone`
4. `sentiment`
5. `gaze_direction`
6. `physical_framing`
7. `activity_engagement`
8. `social_integration`
9. `image_style`

**Not tested in this script** (present in 05/06 but absent here): `perceived_age`,
`environment_type`, `professional_attire`

**Design of comparison:** ALL conditioned images (4 conditions pooled) vs ALL neutral

**BH family size:** 9 p-values

**Total p-values for Nanobanana (charts):** **9**

---

## Script 03_nanobanana_analysis.py

**Operative status:** Currently INOPERATIVE — missing FLUX v2 and SD3.5 v2 input files.
Count below reflects what the code would do if all three CSVs existed.

**Structure:**
```
for col in 9 VISUAL_ATTRIBUTE_COLS:                          # 9 attributes
    for model in [FLUX, SD3.5, Gemini]:                      # 3 models
        for condition in [4 disability conditions]:          # 4 conditions
            → 1 chi2_contingency call
        # BH called on all p-values collected for this col
```

**Tests per attribute:** 3 models × 4 conditions = **12 chi2 calls per attribute**

**BH family per attribute:** 12 p-values (if all 3 models present)

**Total chi2 calls:** 9 attributes × 3 models × 4 conditions = **108 tests**

**BH correction applied:** Per-attribute (9 separate BH calls of 12 p-values each)

**Operative count (Gemini only, since FLUX/SD3.5 missing):**
- If only Gemini model runs: 9 × 1 × 4 = **36 tests**, each BH family of 4

---

## Script 13_explicit_analysis_no_ambiguous.py

**Operative status:** Currently INOPERATIVE (same missing files as above).

Identical structure to `03_nanobanana_analysis.py` with pre-filtering of ambiguous rows.

**Hypothetical full count:** Same as above = **108 tests**, BH per attribute

---

## Script 11_intersectional_analysis.py

**Operative status:** Currently INOPERATIVE (missing FLUX v2 and SD3.5 v2).

**If operative:**
```
for model in [FLUX, SD3.5]:                                  # 2 models
    for condition in [4 disability conditions]:              # 4 conditions
        → 1 chi2_contingency call on Race×Gender intersection
```

**Total chi2 calls:** 2 models × 4 conditions = **8 tests**

**No BH correction.** Raw p < 0.05 threshold.

---

## Script 10_erasure_analysis.py

**No chi2 calls.** Purely descriptive. **0 hypothesis tests.**

---

## Hypothesis Test Summary

| Script | Tests | BH family | BH alpha |
|--------|-------|-----------|---------|
| `05_flux_analysis.py` | **12** | 12 global | 0.05 |
| `06_sd35_analysis.py` | **12** | 12 global | 0.05 |
| `03_nanobanana_charts.py` | **9** | 9 global | 0.05 |
| `03_nanobanana_analysis.py` | 108 (hypothetical) | 12 per attribute | 0.05 |
| `13_explicit_analysis.py` | 108 (hypothetical) | 12 per attribute | 0.05 |
| `11_intersectional_analysis.py` | 8 (hypothetical) | None | N/A |
| `10_erasure_analysis.py` | 0 | None | N/A |

**Operative totals** (scripts that can actually execute given existing files):
- FLUX: 12 tests
- SD3.5: 12 tests
- Nanobanana/Gemini: 9 tests
- **Operative grand total: 33 tests across 3 models**

**Inoperative totals** (scripts that fail due to missing files):
- Tri-model analyses: up to 108 + 108 + 8 = 224 additional tests (all inoperative)

---

## Breakdown Formula (Operative Scripts Only)

```
FLUX:
  12 attributes
  × 1 comparison (all conditions pooled vs neutral)
  × 1 model
  = 12 tests

SD 3.5:
  12 attributes
  × 1 comparison (all conditions pooled vs neutral)
  × 1 model
  = 12 tests

Nanobanana:
  9 attributes
  × 1 comparison (all conditions pooled vs neutral)
  × 1 model
  = 9 tests

OPERATIVE TOTAL = 33
```

---

## Notes on Design

**The pooled design** (all 4 disability conditions merged into a single "conditioned" group) means
that each test in scripts 05, 06, and 03_charts is testing: "Is there ANY distributional difference
between the neutral baseline and the aggregate over all disability conditions?" This is NOT a
per-condition test.

**For per-condition tests**, `03_nanobanana_analysis.py` (and its variants) use a different design
where neutral is compared against each disability condition separately, yielding 4× more tests per
attribute — but these scripts are inoperative.

**Attribute count inconsistency:** Scripts 05/06 test 12 attributes; script 03_charts tests 9.
The three omitted attributes in 03_charts (`perceived_age`, `environment_type`,
`professional_attire`) are present in the FLUX/SD3.5 attribute CSV but absent from the Nanobanana
CSV schema (which has only 18 columns vs 24). This is driven by data schema differences, not
deliberate exclusion.
