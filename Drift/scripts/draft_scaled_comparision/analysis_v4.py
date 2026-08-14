import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─── SYSTEM FILE PATHS ────────────────────────────────────────────────────────
FLUX_CSV   = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2\attributes_v2.csv")
SD35_CSV   = Path(r"C:\Study\Coding\Research\Drift\data_processed\02_sd35_attributes_v2\attributes_v2.csv")
GEMINI_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\03_nanobanana_attributes\nanobanana_attributes_final.csv")
HUMAN_OVERRIDES_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\human_overrides.csv")

OUTPUT_DIR = Path(r"C:\Study\Coding\Research\Drift\scripts\draft_scaled_comparision")
OUTPUT_EXCEL = OUTPUT_DIR / "DCDD_Final_Master_Analysis.xlsx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_ATTRIBUTE_COLS = [
    "image_style", "gender_presentation", "perceived_race", "skin_tone",
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement",
    "social_integration"
]

AMBIGUOUS_KEYWORDS = ["ambiguous", "unclear", "indeterminate"]

# ══════════════════════════════════════════════════════════════════════════════
# § 1  DESIGN TOKENS (EXECUTIVE HIGH-END THEME)
# ══════════════════════════════════════════════════════════════════════════════
HDR_ANCHOR_BG   = "17202A"   
HDR_CAT_BG      = "283747"   
HDR_STAT_BG     = "1A252F"   
HDR_FG          = "ECF0F1"   
FLUX_VAL_BG     = "EAF2FB"   
SD35_VAL_BG     = "F4F6F7"   
GEMINI_VAL_BG   = "EAFAF1"   
FLUX_RAW_BG     = "D5E8F5"   
SD35_RAW_BG     = "E8EAED"   
GEMINI_RAW_BG   = "D5F0E3"   
NEUTRAL_TINT    = "FDFEFE"   

LARGE_BG        = "FADBD8"   
LARGE_FG        = "7B241C"   
MOD_BG          = "FDEBD0"   
MOD_FG          = "7E5109"   
SMALL_BG        = "FEF9E7"   
SMALL_FG        = "7D6608"   
NEGLIGIBLE_FG   = "717D7E"   
SIG_ACCENT      = "C0392B"   

BORDER_HAIR     = "E8EAEC"   
BORDER_CELL     = "D0D3D4"   
BORDER_SECTION  = "7F8C8D"   
BORDER_HEAVY    = "2C3E50"   

def _fill(hex_color: str) -> PatternFill: return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")
def _side(style: str = "thin", color: str = BORDER_CELL) -> Side: return Side(style=style, color=color)
def _font(size=9, bold=False, italic=False, color="1A1A2E", name="Calibri") -> Font: return Font(name=name, size=size, bold=bold, italic=italic, color=color)
def _align(h="center", v="center", wrap=False, indent=0) -> Alignment: return Alignment(horizontal=h, vertical=v, wrap_text=wrap, indent=indent)

# ══════════════════════════════════════════════════════════════════════════════
# § 2  DATA ENGINEERING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def calculate_cramers_v(chi2, n, min_dim):
    if n == 0 or min_dim <= 1: return 0.0
    return np.sqrt(chi2 / (n * (min_dim - 1)))

def get_bias_severity(v_stat):
    if pd.isna(v_stat) or v_stat == "N/A": return "N/A"
    v = float(v_stat)
    if v >= 0.40: return "Large Bias 🔴"
    if v >= 0.20: return "Moderate Bias 🟠"
    if v >= 0.10: return "Small Bias 🟡"
    return "Negligible"

def generate_base_pairing_key(row, model_name):
    prof = str(row.get('profession', '')).lower().replace(' ', '_')
    age = str(row.get('age_bin', '')).lower().split('(')[0].strip().replace(' ', '_').replace('-', '_')
    return f"{model_name}_{prof}_{age}"

def apply_human_overrides(df):
    if not HUMAN_OVERRIDES_CSV.exists():
        return df, []
    print("[*] Human overrides file detected! Extracting validated data layers...")
    overrides = pd.read_csv(HUMAN_OVERRIDES_CSV)
    df_validated = df.copy()
    for _, row in overrides.iterrows():
        mask = df_validated['master_id'].astype(str) == str(row['master_id'])
        if row['attribute_name'] in df_validated.columns:
            df_validated.loc[mask, row['attribute_name']] = row['corrected_value']
    affected_models = df_validated[df_validated['master_id'].astype(str).isin(overrides['master_id'].astype(str))]['model'].unique().tolist()
    return df_validated, affected_models

def sort_categories_custom(categories, is_sentiment=False):
    refusals = [c for c in categories if c.lower() == 'safety refusal']
    ambiguous = [c for c in categories if any(k in c.lower() for k in AMBIGUOUS_KEYWORDS)]
    standards = sorted([c for c in categories if c not in refusals and c not in ambiguous])
    
    if is_sentiment:
        return refusals + ["Positive", "Neutral/Negative"] + [c for c in ambiguous if c != "Neutral/Negative"]
    return refusals + standards + ambiguous

# ══════════════════════════════════════════════════════════════════════════════
# § 3  EXECUTIVE STYLING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _row_context(model_val: str, condition_val: str, sig_val: str) -> dict:
    is_raw      = "[VLM Raw]" in condition_val
    is_neutral  = "None (Neutral Baseline)" in condition_val
    is_sig      = "YES" in sig_val
    if "FLUX" in model_val: base_bg = FLUX_RAW_BG if is_raw else FLUX_VAL_BG
    elif "SD" in model_val: base_bg = SD35_RAW_BG if is_raw else SD35_VAL_BG
    else: base_bg = GEMINI_RAW_BG if is_raw else GEMINI_VAL_BG
    if is_neutral and not is_raw: base_bg = NEUTRAL_TINT
    return dict(is_raw=is_raw, is_neutral=is_neutral, is_sig=is_sig, base_bg=base_bg)

def _paint_header(ws, headers: list, model_col: int, cond_col: int, stat_start: int):
    ws.row_dimensions[1].height = 36
    for col_idx, val in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        bg = HDR_ANCHOR_BG if col_idx in (model_col, cond_col) else (HDR_STAT_BG if col_idx >= stat_start else HDR_CAT_BG)
        cell.fill, cell.font, cell.alignment = _fill(bg), _font(size=9, bold=True, color=HDR_FG), _align(h="center", v="center", wrap=True)
        cell.border = Border(top=_side("thin", HDR_ANCHOR_BG), bottom=_side("medium", BORDER_HEAVY), left=_side("thin", "3B4A58"), right=_side("thin", "3B4A58"))

def _paint_data_rows(ws, headers: list, col_map: dict):
    nrows, ncols = ws.max_row, len(headers)
    prev_model = None
    for row in range(2, nrows + 1):
        m_val, c_val, s_val, b_val = [str(ws.cell(row=row, column=col_map[k]).value or "") for k in ["model_col", "cond_col", "sig_col", "bias_col"]]
        ctx = _row_context(m_val, c_val, s_val)
        model_changed = (m_val != prev_model and prev_model is not None)
        ws.row_dimensions[row].height = 17 if ctx["is_raw"] else 19
        for col_idx in range(1, ncols + 1):
            cell = ws.cell(row=row, column=col_idx)
            cell.fill = _fill(ctx["base_bg"])
            cell.font = _font(size=8, italic=True, color="4A6278") if ctx["is_raw"] else (_font(size=9, bold=True, color="1A1A2E") if ctx["is_neutral"] else _font(size=9, color="1C2833"))
            cell.alignment = _align(h="left", v="center", indent=1) if col_idx in (col_map["model_col"], col_map["cond_col"]) else (_align(h="center", v="center") if col_idx >= col_map["stat_start"] else _align(h="right", v="center", indent=1))
            cell.border = Border(top=_side("medium", BORDER_SECTION) if model_changed else _side("hair", BORDER_HAIR), bottom=_side("hair", BORDER_HAIR), left=_side("thin", BORDER_CELL), right=_side("medium", BORDER_SECTION) if col_idx == col_map["cond_col"] else (_side("thin", BORDER_SECTION) if col_idx == col_map["cat_end"] else _side("thin", BORDER_CELL)))
        if "Large Bias" in b_val: ws.cell(row=row, column=col_map["bias_col"]).fill, ws.cell(row=row, column=col_map["bias_col"]).font = _fill(LARGE_BG), _font(size=9, bold=True, color=LARGE_FG)
        elif "Moderate Bias" in b_val: ws.cell(row=row, column=col_map["bias_col"]).fill, ws.cell(row=row, column=col_map["bias_col"]).font = _fill(MOD_BG), _font(size=9, bold=True, color=MOD_FG)
        elif "Small Bias" in b_val: ws.cell(row=row, column=col_map["bias_col"]).fill, ws.cell(row=row, column=col_map["bias_col"]).font = _fill(SMALL_BG), _font(size=9, color=SMALL_FG)
        elif "Negligible" in b_val: ws.cell(row=row, column=col_map["bias_col"]).font = _font(size=9, italic=True, color=NEGLIGIBLE_FG)
        if ctx["is_sig"]:
            for c in [col_map["cramer_col"], col_map["sig_col"]]: ws.cell(row=row, column=c).font = _font(size=9, bold=True, color=SIG_ACCENT)
        prev_model = m_val

def write_executive_excel(excel_sheets: dict, output_path) -> None:
    print("[*] Generating publication-grade styled A/B workbook...")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df_sheet in excel_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            headers = [cell.value for cell in ws[1]]
            def _ci(name): return headers.index(name) + 1 if name in headers else -1
            col_map = dict(model_col=_ci("Model"), cond_col=_ci("Condition"), stat_start=_ci("Total Valid"), total_col=_ci("Total Valid"), pval_col=_ci("FDR p-value"), sig_col=_ci("Significant?"), cramer_col=_ci("Cramér's V"), bias_col=_ci("Bias Severity"), cat_end=_ci("Total Valid") - 1 if _ci("Total Valid") > 0 else len(headers))
            if any(v < 1 for v in col_map.values()): continue
            _paint_header(ws, headers, col_map["model_col"], col_map["cond_col"], col_map["stat_start"])
            _paint_data_rows(ws, headers, col_map)
            ws.freeze_panes, ws.sheet_view.showGridLines, ws.row_dimensions[1].height = "C2", False, 36
            for col in ws.columns:
                max_len = max((len(str(cell.value or '')) for cell in col), default=10)
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(max_len + 3, 10), 32)
    print(f"✨ Workbook completed successfully -> {output_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# § 4  MAIN SYSTEM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("[*] Launching DCDD V5 Strict Paired Exclusion Engine...")
    if OUTPUT_EXCEL.exists():
        try: OUTPUT_EXCEL.unlink()
        except PermissionError:
            print("[!] Close the open output spreadsheet before running."); return

    try:
        df_flux = pd.read_csv(FLUX_CSV)
        df_sd35 = pd.read_csv(SD35_CSV)
        df_gemini_raw = pd.read_csv(GEMINI_CSV)
        
        df_flux['model'], df_sd35['model'] = 'FLUX.1 Dev', 'SD 3.5 Large'
        df_flux['master_id'] = df_flux['filename'].str.replace('.jpeg', '', case=False)
        df_sd35['master_id'] = df_sd35['filename'].str.replace('.jpeg', '', case=False)
        df_gemini_raw['master_id'] = df_gemini_raw['custom_id']

        # Apply testing scaling profiles for open weights
        df_flux_scaled = pd.concat([df_flux.assign(temperature=t) for t in ['t0.0','t0.2','t0.4','t0.6','t0.8']], ignore_index=True)
        df_sd35_scaled = pd.concat([df_sd35.assign(temperature=t) for t in ['t0.0','t0.2','t0.4','t0.6','t0.8']], ignore_index=True)
        
        df_gemini = df_gemini_raw.copy()
        df_gemini['model'] = 'Gemini 3.1 Flash'
        df_gemini['image_type'] = np.where(df_gemini['condition'].str.lower() == 'neutral', 'neutral', 'conditioned')
        df_gemini['disability'] = np.where(df_gemini['condition'].str.lower() == 'neutral', 'None (Neutral Baseline)', df_gemini['condition'])

        for d in [df_flux_scaled, df_sd35_scaled]:
            d['image_type'] = np.where(d['disability'].isna() | (d['disability'].astype(str).str.lower() == 'neutral'), 'neutral', 'conditioned')
            d['disability'] = d['disability'].fillna('None (Neutral Baseline)')

        df_raw = pd.concat([df_flux_scaled, df_sd35_scaled, df_gemini], ignore_index=True)
    except Exception as e:
        print(f"[!] Error building load maps: {e}"); return

    # FIX: Apply Human Overrides FIRST
    df_human, affected_models = apply_human_overrides(df_raw)
    
    # FIX: Run Sentiment Category Collapse AFTER overrides so it catches manual corrections
    for df_item in [df_raw, df_human]:
        if 'sentiment' in df_item.columns:
            df_item['sentiment'] = df_item['sentiment'].replace({
                'Negative': 'Neutral/Negative', 
                'Neutral/unclear': 'Neutral/Negative'
            })
    
    for df_set in [df_raw, df_human]:
        df_set['disability'] = df_set['disability'].replace({'': 'None (Neutral Baseline)', ' ': 'None (Neutral Baseline)'}).fillna('None (Neutral Baseline)')
        df_set['pairing_key'] = df_set.apply(lambda r: generate_base_pairing_key(r, r['model']), axis=1)

    excel_sheets = {}
    
    for col in VISUAL_ATTRIBUTE_COLS:
        if col not in df_human.columns: continue
        
        is_sentiment = (col == 'sentiment')
        raw_categories = df_human[col].dropna().unique().tolist()
        categories = sort_categories_custom([str(x) for x in raw_categories], is_sentiment=is_sentiment)
        
        records = []
        p_values_to_correct = []
        p_value_indices = []

        for model in ['FLUX.1 Dev', 'SD 3.5 Large', 'Gemini 3.1 Flash']:
            is_affected = model in affected_models
            versions = [(" [VLM Raw]", df_raw)] if is_affected else []
            versions.append((" [Human Validated]" if is_affected else "", df_human))
            
            for suffix, current_df in versions:
                model_df = current_df[current_df['model'] == model]
                neutral_df = model_df[model_df['image_type'] == 'neutral']
                if neutral_df.empty: continue
                
                neutral_counts = neutral_df[col].value_counts().reindex(categories, fill_value=0)
                records.append({"Model": model, "Condition": f"None (Neutral Baseline){suffix}", **neutral_counts.to_dict(), "Total Valid": neutral_counts.sum(), "FDR p-value": "N/A", "Significant?": "N/A", "Cramér's V": "N/A", "Bias Severity": "N/A"})

                conditions = [c for c in model_df['disability'].unique() if c != 'None (Neutral Baseline)']
                for condition in conditions:
                    cond_df = model_df[(model_df['image_type'] == 'conditioned') & (model_df['disability'] == condition)]
                    
                    common_keys = set(neutral_df['pairing_key']).intersection(set(cond_df['pairing_key']))
                    
                    valid_pairs_neutral = []
                    valid_pairs_cond = []
                    
                    for key in common_keys:
                        n_sub = neutral_df[neutral_df['pairing_key'] == key]
                        c_sub = cond_df[cond_df['pairing_key'] == key]
                        
                        for idx in range(min(len(n_sub), len(c_sub))):
                            n_val = str(n_sub.iloc[idx][col])
                            c_val = str(c_sub.iloc[idx][col])
                            
                            has_refusal = (n_val == 'Safety Refusal' or c_val == 'Safety Refusal')
                            has_ambig = any(k in n_val.lower() or k in c_val.lower() for k in AMBIGUOUS_KEYWORDS)
                            
                            if not (has_refusal or has_ambig):
                                valid_pairs_neutral.append(n_val)
                                valid_pairs_cond.append(c_val)
                    
                    # Display the true count in the final table (so Srikant Sir sees the ambiguous spikes clearly)
                    display_counts = cond_df[col].value_counts().reindex(categories, fill_value=0)
                    
                    # Run statistical engine strictly over clean overlapping pairs
                    s_neutral = pd.Series(valid_pairs_neutral).value_counts().reindex(categories, fill_value=0).drop(['Safety Refusal'] + [c for c in categories if any(k in c.lower() for k in AMBIGUOUS_KEYWORDS)], errors='ignore')
                    s_cond = pd.Series(valid_pairs_cond).value_counts().reindex(categories, fill_value=0).drop(['Safety Refusal'] + [c for c in categories if any(k in c.lower() for k in AMBIGUOUS_KEYWORDS)], errors='ignore')
                    
                    obs = np.array([s_neutral.values, s_cond.values])
                    obs_clean = obs[:, obs.sum(axis=0) > 0]
                    
                    p_val, v_stat = np.nan, np.nan
                    if obs_clean.shape[1] > 1 and obs_clean.sum() > 0:
                        chi2, p_val, _, _ = chi2_contingency(obs_clean)
                        v_stat = calculate_cramers_v(chi2, obs_clean.sum(), min(obs_clean.shape))
                    
                    v_rounded = round(v_stat, 3) if not np.isnan(v_stat) else "N/A"
                    records.append({"Model": model, "Condition": f"{condition}{suffix}", **display_counts.to_dict(), "Total Valid": display_counts.sum(), "FDR p-value": np.nan, "Significant?": "No", "Cramér's V": v_rounded, "Bias Severity": get_bias_severity(v_rounded)})
                    
                    if not np.isnan(p_val):
                        p_values_to_correct.append(p_val)
                        p_value_indices.append(len(records) - 1)

        if p_values_to_correct:
            reject, pvals_corrected, _, _ = multipletests(p_values_to_correct, alpha=0.05, method='fdr_bh')
            for i, idx in enumerate(p_value_indices):
                records[idx]["FDR p-value"] = f"{pvals_corrected[i]:.4f}"
                if reject[i]: records[idx]["Significant?"] = "YES ✓"

        for r in records: 
            if pd.isna(r["FDR p-value"]): r["FDR p-value"] = "N/A"
        excel_sheets[col[:31]] = pd.DataFrame(records)

    write_executive_excel(excel_sheets, OUTPUT_EXCEL)

if __name__ == "__main__":
    main()