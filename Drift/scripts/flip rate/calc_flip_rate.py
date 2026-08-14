import os
import pandas as pd
import numpy as np

# --- ENVIRONMENT PATH ARCHITECTURE ---
PATHS = {
    "Gemini_Flash_3.1": r"C:\Study\Coding\Research\Drift\data_processed\03_nanobanana_attributes\nanobanana_attributes_v2_validated.csv",
    "FLUX.1_Dev": r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2\attributes_v2.csv",
    "Stable_Diffusion_3.5_Large": r"C:\Study\Coding\Research\Drift\data_processed\02_sd35_attributes_v2\attributes_v2.csv"
}

OUTPUT_DIR = r"C:\Study\Coding\Research\Drift\scripts\flip rate"

ATTRIBUTES = [
    "image_style", "gender_presentation", "perceived_race", "skin_tone", 
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement", "social_integration"
]

def calculate_model_flip_rates(model_name, file_path):
    if not os.path.exists(file_path):
        print(f"[!] Target data file absent for {model_name} at: {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    
    # Resolve structural schema variances dynamically
    type_col = 'image_type' if 'image_type' in df.columns else 'condition'
    disability_col = 'disability' if 'disability' in df.columns else 'condition'
    
    # 1. Isolate the reference cases (Neutral Baseline prompts)
    neutral_mask = (df[type_col].astype(str).str.lower() == 'neutral') | \
                   (df[disability_col].astype(str).str.lower().str.contains('none|baseline|neutral', na=False))
    
    df_neutral = df[neutral_mask].copy()
    df_conditioned = df[~neutral_mask].copy()
    
    # Build the reference map from baseline matrices
    baseline_map = {}
    for _, row in df_neutral.iterrows():
        pid = row['pair_id']
        baseline_map[pid] = {attr: str(row[attr]).strip().lower() for attr in ATTRIBUTES if attr in df.columns}
        
    # Standardize qualitative condition text cells
    def clean_condition(val):
        val = str(val).lower().strip()
        if 'mobility' in val: return 'Mobility Impairment'
        if 'vision' in val: return 'Vision Impairment'
        if 'hearing' in val: return 'Hearing Impairment'
        if 'mental' in val or 'psych' in val: return 'Mental Health Condition'
        return 'Other'

    df_conditioned['clean_disability'] = df_conditioned[disability_col].apply(clean_condition)
    conditions = ['Mobility Impairment', 'Vision Impairment', 'Hearing Impairment', 'Mental Health Condition']
    
    # Build execution tracking matrix
    results = {cond: {attr: [] for attr in ATTRIBUTES} for cond in conditions}
    
    # 2. Iterate across counterfactual cells to detect flips
    for _, row in df_conditioned.iterrows():
        pid = row['pair_id']
        cond = row['clean_disability']
        
        if cond not in conditions or pid not in baseline_map:
            continue
            
        # Drop hard safety refusals from active statistical accumulation
        if str(row[ATTRIBUTES[0]]).strip().lower() == 'safety refusal':
            continue
            
        base_line = baseline_map[pid]
        
        for attr in ATTRIBUTES:
            if attr not in row or attr not in base_line:
                continue
            
            curr_val = str(row[attr]).strip().lower()
            base_val = base_line[attr]
            
            if 'safety refusal' in curr_val or 'safety refusal' in base_val:
                continue
                
            # Binary switch check: 1 implies label alteration, 0 implies parity
            is_flipped = 1 if curr_val != base_val else 0
            results[cond][attr].append(is_flipped)
            
    # Compile execution data frame blocks
    final_table = pd.DataFrame(index=ATTRIBUTES)
    for cond in conditions:
        cond_rates = []
        for attr in ATTRIBUTES:
            arr = results[cond][attr]
            rate = (sum(arr) / len(arr) * 100) if len(arr) > 0 else 0.0
            cond_rates.append(rate)
        final_table[cond] = cond_rates
        
    return final_table

def main():
    # Guarantee output folder verification boundary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[*] Root Output Location Initialized: {OUTPUT_DIR}")
    
    for model_name, path in PATHS.items():
        print(f"\n[+] Processing profile matrices for framework: {model_name}...")
        res_df = calculate_model_flip_rates(model_name, path)
        
        if res_df is not None:
            # Round numerical elements to meet strict review documentation formatting
            final_df = res_df.round(2)
            
            # 1. Terminal screen print visualization
            print(final_df.to_string())
            print("-" * 65)
            
            # 2. Disk compilation and execution safe save
            filename = f"flip_rates_{model_name.lower()}.csv"
            save_path = os.path.join(OUTPUT_DIR, filename)
            final_df.to_csv(save_path)
            print(f"[->] Successfully exported file matrix to: {save_path}")

if __name__ == "__main__":
    main()