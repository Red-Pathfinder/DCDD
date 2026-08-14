import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path
from openpyxl.styles import PatternFill, Font

# ─── PATHS ──────────────────────────────────────────────────────────────────────
FLUX_ATTR_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2\attributes_v2.csv")
SD35_ATTR_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\02_sd35_attributes_v2\attributes_v2.csv")

OUTPUT_DIR = Path(r"C:\Study\Coding\Research\Drift\results")
OUTPUT_EXCEL = OUTPUT_DIR / "DCDD_Intersectional_Bias_Results.xlsx"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── COLORS FOR SPREADSHEET ─────────────────────────────────────────────────────
HEADER_FILL = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
SIGNIFICANT_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid") # Red for p < 0.05

def calculate_cramers_v(confusion_matrix):
    """Calculates Cramér's V effect size from a contingency table."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    if n == 0 or min(r, k) < 2: return 0.0
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def main():
    print("[*] Loading attribute data...")
    
    # Load and merge the two datasets
    try:
        flux_df = pd.read_csv(FLUX_ATTR_CSV)
        flux_df['model'] = 'FLUX.1 Dev'
        
        sd35_df = pd.read_csv(SD35_ATTR_CSV)
        sd35_df['model'] = 'SD 3.5 Large'
        
        df = pd.concat([flux_df, sd35_df], ignore_index=True)
        print(f"[*] Successfully merged {len(df)} total image records.")
    except FileNotFoundError as e:
        print(f"[!] Error: Could not find one of the CSV files. Details: {e}")
        return

    print("[*] Creating Intersectional Categories (Race + Gender)...")
    
    # Clean up the strings to make them readable (e.g., "White presenting" -> "White")
    # Using .astype(str) to handle any NaN values gracefully
    df['race_clean'] = df['perceived_race'].astype(str).str.replace(' presenting', '', regex=False).str.strip()
    df['gender_clean'] = df['gender_presentation'].astype(str).str.replace(' presenting', '', regex=False).str.strip()
    
    # FUSE THEM: e.g., "White - Masculine"
    df['Intersection'] = df['race_clean'] + " - " + df['gender_clean']

    results = []

    for model in ['FLUX.1 Dev', 'SD 3.5 Large']:
        model_df = df[df['model'] == model]
        if model_df.empty: continue

        neutral_df = model_df[model_df['image_type'] == 'neutral']
        
        # Ensure 'disability' column exists, otherwise fall back to 'condition'
        condition_col = 'disability' if 'disability' in model_df.columns else 'condition'
        
        if condition_col not in model_df.columns:
            print(f"[!] Warning: Could not find 'disability' or 'condition' column in {model} data.")
            continue
            
        conditions = [d for d in model_df[condition_col].unique() if pd.notna(d) and str(d).strip() != 'neutral']

        for condition in conditions:
            # We assume 'conditioned' image_type, but if that column doesn't exist, we filter by condition name
            cond_df = model_df[model_df[condition_col] == condition]
            
            if neutral_df.empty or cond_df.empty:
                continue
                
            # Count the intersections
            neutral_counts = neutral_df['Intersection'].value_counts()
            cond_counts = cond_df['Intersection'].value_counts()

            # --- THE FIX: Calculate TRUE Shift using Percentages ---
            neutral_pct = neutral_counts / neutral_counts.sum()
            cond_pct = cond_counts / cond_counts.sum()
            
            # Calculate the delta (which category grew the most?)
            delta = cond_pct.subtract(neutral_pct, fill_value=0)
            
            if not delta.empty:
                true_shift_cat = delta.idxmax()
                shift_magnitude = delta.max() * 100 
            else:
                true_shift_cat = 'N/A'
                shift_magnitude = 0.0

            # Align categories for Chi-Square Matrix
            all_categories = list(set(neutral_counts.index).union(set(cond_counts.index)))
            
            obs = []
            for cat in all_categories:
                obs.append([neutral_counts.get(cat, 0), cond_counts.get(cat, 0)])
            
            # We need at least a 2x2 table to run Chi-Square
            if len(obs) > 1 and sum(sum(row) for row in obs) > 0:
                chi2, p_val, dof, expected = chi2_contingency(obs)
                c_v = calculate_cramers_v(pd.DataFrame(obs))
            else:
                p_val, c_v = 1.0, 0.0

            results.append({
                "Model": model,
                "Disability Introduced": condition.title(),
                "P-Value": p_val,
                "Significant (p < 0.05)": "YES" if p_val < 0.05 else "NO",
                "Effect Size (Cramér's V)": c_v,
                "True Intersectional Shift": f"Grew by +{shift_magnitude:.1f}% towards {true_shift_cat}"
            })

    summary_df = pd.DataFrame(results)

    print("[*] Saving Intersectional Findings...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Intersectional Bias", index=False)
        worksheet = writer.sheets["Intersectional Bias"]
        
        # Formatting Headers
        for cell in worksheet[1]:
            cell.fill = HEADER_FILL
            cell.font = Font(color="FFFFFF", bold=True)
            
        # Auto-adjust column widths
        for excel_col in worksheet.columns:
            max_length = max((len(str(cell.value)) for cell in excel_col), default=0)
            worksheet.column_dimensions[excel_col[0].column_letter].width = min(max_length + 2, 45)

        # Highlight significant rows
        for row_idx, row in enumerate(summary_df.itertuples(), start=2):
            if row._4 == "YES":  # Index 4 is the "Significant (p < 0.05)" column
                for col_idx in range(1, 7):
                    worksheet.cell(row=row_idx, column=col_idx).fill = SIGNIFICANT_FILL
                    worksheet.cell(row=row_idx, column=col_idx).font = Font(color="9C0006", bold=True)

    print(f"✨ DONE! Check {OUTPUT_EXCEL} for the true intersectional shifts!")

if __name__ == "__main__":
    main()

