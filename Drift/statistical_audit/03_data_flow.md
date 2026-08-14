# 03 — Data Flow

> **Forensic Audit of the DCDD Repository**
> Every value verified from CSV files and source code.

---

## Model: FLUX

### Which CSV is read

**Script:** `05_flux_analysis.py`, line 37:
```python
DATA_PATH = BASE_DIR / "Drift" / "data_processed" / "01_flux_attributes" / "attributes.csv"
```

**File on disk:** `data_processed/01_flux_attributes/attributes.csv` — **EXISTS** ✅

**BASE_DIR resolution** (line 33): `Path(__file__).resolve().parents[3]`
Script is at `…/Drift/scripts/03_analysis/`, so parents[3] = `C:\Study\Coding\Research`.
Full resolved path: `C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes\attributes.csv`

---

### CSV schema (verified by inspection)

| Column | Type | Notes |
|--------|------|-------|
| `dcdd_id` | str | Unique image identifier |
| `pair_id` | str | Pairing identifier |
| `seed` | int | Generation seed |
| `model` | str | Model name |
| `sector` | str | Occupational sector |
| `profession` | str | Specific profession |
| `age_bin` | str | Age category |
| `image_type` | str | `neutral` or `conditioned` |
| `disability` | str | Disability condition or NaN for neutral |
| `prompt` | str | Full prompt text |
| `filename` | str | Image filename |
| `timestamp` | str | Generation time |
| `image_style` | str | VLM-extracted attribute |
| `gender_presentation` | str | VLM-extracted attribute |
| `perceived_race` | str | VLM-extracted attribute |
| `skin_tone` | str | VLM-extracted attribute |
| `perceived_age` | str | VLM-extracted attribute |
| `sentiment` | str | VLM-extracted attribute |
| `environment_type` | str | VLM-extracted attribute |
| `professional_attire` | str | VLM-extracted attribute |
| `gaze_direction` | str | VLM-extracted attribute |
| `physical_framing` | str | VLM-extracted attribute |
| `activity_engagement` | str | VLM-extracted attribute |
| `social_integration` | str | VLM-extracted attribute |

**Total columns:** 24 (12 metadata + 12 VLM attribute columns)

---

### Raw row counts (before any filtering)

| image_type | rows |
|------------|------|
| conditioned | 305 |
| neutral | 75 |
| **Total** | **380** |

### age_bin distribution (before teen filter)

| age_bin | count |
|---------|-------|
| adult (26-40) | 129 |
| senior (60+) | 128 |
| young adult (18-25) | 123 |
| teen (13-17) | **0** |

**Finding:** No `teen (13-17)` rows exist in `attributes.csv` at time of inspection. The
`00_purge_flux_teens.py` script was already run and removed teen rows from the CSV. The teen filter
in `load_data()` (`df = df[df['age_bin'] != 'teen (13-17)']`, line 79) is therefore a **no-op**
at runtime — it removes 0 rows.

---

### Is a validated dataset used?

**No.** The script reads directly from `01_flux_attributes/attributes.csv`.
There is no merge with `human_overrides.csv` anywhere in `05_flux_analysis.py`.

**human_overrides.csv** (`data_processed/human_overrides.csv`) contains 1,703 rows with columns
`[master_id, attribute_name, corrected_value]`. It is written by the human annotation tools
(`08_human_audit.py`, `human_validation_of_sd_flux_non_amb.py`) but **never read by the
statistical analysis scripts** (05_flux_analysis.py, 06_sd35_analysis.py, or any other analysis
script except indirectly via a possible pre-baked merge into the CSV — see below).

**Are human corrections baked into the CSV?** Unknown from code alone. The attributes.csv file
on disk could have been manually updated. The code does not do it programmatically within the
analysis scripts.

---

### Are raw VLM outputs used?

The `attributes.csv` file is the output of the VLM batch extraction. Whether human overrides were
applied to it before it was saved to disk cannot be determined from the analysis scripts alone.
The analysis scripts read what is in `attributes.csv` as-is.

---

### Are ambiguous labels removed?

**For chi-square:** No. `pd.crosstab(df['image_type'], df[attribute])` includes all rows. Ambiguous
labels appear as columns in the contingency table.

**For CFR:** Yes — the CFR function explicitly skips pairs where either value is `'ambiguous'`:
```python
if n_val != 'ambiguous' and c_val != 'ambiguous':
```

**Ambiguous counts in FLUX v1 attributes.csv:**
- `gender_presentation`: 20 rows contain "Ambiguous"
- `perceived_race`: 96 rows contain "Ambiguous"
- All other attributes: 0 ambiguous rows

---

### Are Safety Refusal rows removed?

**Not applicable to FLUX.** No rows in FLUX attributes.csv contain "Safety Refusal" in any column.

---

### Are missing values dropped?

No explicit `dropna()` is called on the data. `pd.crosstab()` ignores NaN values by default.
The `dropna(how='all')` on line 130 applies to rows of the cross-tabulation table (i.e., removes
entire condition groups that are entirely NaN), not to the input data rows.

---

### Are rows duplicated?

No intentional duplication. However, the conditioned set (305 rows) contains 4 disability
conditions sharing the same 75–77 base prompt seeds. Pair_key analysis:
- Unique neutral pair_keys: 75
- Unique conditioned pair_keys: 77 (2 extra conditioned seeds have no neutral match)
- Conditioned pair_keys are duplicated (each seed appears ~4×) — this is by experimental design,
  not a data error
- For the chi-square, all 305 conditioned rows are treated as independent observations

---

### How is N computed?

For chi-square: `N = contingency_table.sum()` inside `cramers_v()` = total cells in the 2×K table
= 380 rows (75 neutral + 305 conditioned).

For CFR: `total_pairs` counts matched pairs where neither value is 'ambiguous'. Maximum possible
value ≤ (number of conditioned rows with a matching neutral pair) × 1. Since 75 neutral keys and
77 conditioned unique keys exist with 75 overlapping, maximum matched pairs = 75 × (number of
conditions that row appears in as conditioned) ≤ 305.

---

---

## Model: SD 3.5

### Which CSV is read

**Script:** `06_sd35_analysis.py`, line 36:
```python
DATA_PATH = BASE_DIR / "Drift" / "data_processed" / "02_sd35_attributes" / "attributes.csv"
```

**File on disk:** `data_processed/02_sd35_attributes/attributes.csv` — **EXISTS** ✅

### Raw row counts

| image_type | rows |
|------------|------|
| conditioned | 308 |
| neutral | 77 |
| **Total** | **385** |

### age_bin distribution

| age_bin | count |
|---------|-------|
| adult (26-40) | 130 |
| senior (60+) | 130 |
| young adult (18-25) | 125 |
| teen (13-17) | **0** |

Teen filter is again a no-op (0 teen rows in CSV).

### Columns

Identical 24-column schema as FLUX. All 12 VLM attribute columns present.

### Are ambiguous labels removed?

For chi-square: No (same behavior as FLUX).
For CFR: Yes (same `!= 'ambiguous'` check).

### Are Safety Refusal rows removed?

No Safety Refusal entries in SD3.5 data (not present in any column).

---

---

## Model: Nanobanana (Gemini)

### Script A — 03_nanobanana_charts.py (primary per-model analysis)

**CSV read** (line 21):
```
data_processed/03_nanobanana_attributes/nanobanana_attributes_final.csv
```
**EXISTS** ✅

**Shape:** 1,925 rows × 18 columns

**Columns (18):** `custom_id, pair_id, sector, profession, age_bin, condition, temperature, seed,
prompt, image_style, gender_presentation, perceived_race, skin_tone, sentiment, gaze_direction,
physical_framing, activity_engagement, social_integration`

**Key difference from FLUX/SD3.5:**
- No `image_type` column in raw data
- No `disability` column in raw data
- `condition` column holds values: `neutral`, `mobility impairment`, `vision impairment`,
  `hearing impairment`, `mental health condition`
- `load_and_harmonize()` derives `image_type` as: `neutral` if condition == 'neutral' else `conditioned`

**Condition distribution:**
| condition | rows |
|-----------|------|
| neutral | 385 |
| mobility impairment | 385 |
| vision impairment | 385 |
| hearing impairment | 385 |
| mental health condition | 385 |
| **Total** | **1,925** |

**Chi-square N:** 1,925 (385 neutral + 1,540 conditioned, all 4 conditions pooled)

**Are Safety Refusal rows removed?** No — they are included as a category in the chi-square table.
218 rows have `image_style == 'Safety Refusal'` across all columns for those rows.

**Safety Refusal by condition:**
| condition | refusal count |
|-----------|--------------|
| mental health condition | 145 |
| hearing impairment | 24 |
| vision impairment | 19 |
| mobility impairment | 17 |
| neutral | 13 |

**Are ambiguous labels removed?** No — for chi-square in `test_attribute()`, all rows including
ambiguous and Safety Refusal are included. The `calculate_flip_rate()` function is **not called**
in `03_nanobanana_charts.py` (no CFR computed for Nanobanana).

---

### Script B — 03_nanobanana_analysis.py (tri-model Excel output)

**CSVs read** (lines 9–11):
- `01_flux_attributes_v2/attributes_v2.csv` → **MISSING** ❌
- `02_sd35_attributes_v2/attributes_v2.csv` → **MISSING** ❌
- `03_nanobanana_attributes/nanobanana_attributes_final.csv` → **EXISTS** ✅

The try/except at lines 56–69 will catch the `FileNotFoundError` for the missing CSVs and `return`
without producing any output. The Excel file `DCDD_Final_Analysis_with_bias_severity.xlsx` will
**not be produced** if the FLUX and SD3.5 v2 CSVs are absent.

**Note on v2 CSVs:** The actual v2 files on disk are:
- `01_flux_attributes_v2/flux_attributes_v2.csv` (380 rows, 21 cols, no `perceived_age`/`environment_type`/`professional_attire`)
- `02_sd35_attributes_v2/sd_attributes_v2.csv` (385 rows, 21 cols, same)

These files have different filenames from what the script expects.

---

### Script C — 13_explicit_analysis_no_ambiguous.py

Same path issue as Script B — will fail on missing FLUX/SD3.5 v2 CSVs.

**Exclusion logic (when operative):**
```python
EXCLUSION_KEYWORDS = ['ambiguous', 'unclear', 'indeterminate', 'safety refusal']
explicit_mask = df[col].apply(is_explicit)
df_explicit = df[explicit_mask].copy()
```
Rows are excluded per-attribute (only rows where the current attribute column is explicit are
retained). A row excluded for `gender_presentation` may still appear in the `perceived_race` test.

---

### Script D — 11_intersectional_analysis.py

**CSVs read** (lines 8–9):
- `01_flux_attributes_v2/attributes_v2.csv` → **MISSING** ❌
- `02_sd35_attributes_v2/attributes_v2.csv` → **MISSING** ❌

Will fail at `pd.read_csv()`. The try/except at lines 33–44 catches `FileNotFoundError` and
`return`s. No intersectional output will be produced.

---

## Cross-Model Summary Table

| Property | FLUX | SD 3.5 | Nanobanana (charts) |
|----------|------|--------|---------------------|
| CSV used | `01_flux_attributes/attributes.csv` | `02_sd35_attributes/attributes.csv` | `03_nanobanana_attributes/nanobanana_attributes_final.csv` |
| CSV exists | ✅ | ✅ | ✅ |
| Validated dataset | Not explicit | Not explicit | Not explicit |
| Human overrides applied in script | ❌ No | ❌ No | ❌ No |
| Raw VLM outputs | Yes (as-is from CSV) | Yes | Yes |
| Ambiguous in chi-square | ✅ Included | ✅ Included | ✅ Included |
| Ambiguous excluded for CFR | ✅ Yes | ✅ Yes | N/A (no CFR) |
| Safety Refusal rows | Not present | Not present | ✅ Included |
| Missing values dropped | No explicit drop | No explicit drop | No explicit drop |
| Rows duplicated intentionally | No | No | No |
| N for chi-square | 380 | 385 | 1,925 |
| N neutral | 75 | 77 | 385 |
| N conditioned (pooled) | 305 | 308 | 1,540 |
| Teen filter | No-op (0 teens) | No-op (0 teens) | Not applied |
