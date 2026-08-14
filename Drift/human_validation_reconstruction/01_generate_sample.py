import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(r"C:\Study\Coding\Research\DCDD\Drift")
OUTPUT_DIR = BASE_DIR / "human_validation_reconstruction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
FLUX_CSV = BASE_DIR / "data_processed" / "01_flux_attributes_v2" / "flux_attributes_v2.csv"
SD35_CSV = BASE_DIR / "data_processed" / "02_sd35_attributes_v2" / "sd_attributes_v2.csv"
NANO_CSV = BASE_DIR / "data_processed" / "03_nanobanana_attributes" / "nanobanana_attributes_final.csv"

# Image paths
FLUX_IMG_DIR = BASE_DIR / "data_raw" / "flux_pipeline"
SD35_IMG_DIR = BASE_DIR / "data_raw" / "sd35_pipeline"
NANO_IMG_DIR = BASE_DIR / "data_raw" / "nanobanana_2_pipeline"

# 6 final attributes
ATTRIBUTES = [
    "gender_presentation", "skin_tone", "sentiment",
    "physical_framing", "activity_engagement", "social_integration"
]

def map_condition(val):
    if pd.isna(val): return "Neutral"
    val = str(val).lower()
    if "neutral" in val or val.strip() == "": return "Neutral"
    if "mobility" in val: return "Mobility"
    if "vision" in val: return "Vision"
    if "hearing" in val: return "Hearing"
    if "mental" in val: return "Mental Health"
    return "Other"

def sample_dataset(df, n_samples, id_col, model_name, img_dir, img_ext):
    valid_rows = []
    for idx, row in df.iterrows():
        img_id_val = str(row[id_col])
        if 'filename' in row and pd.notna(row['filename']):
            img_path = img_dir / str(row['filename'])
            actual_id = str(row['filename'])
        else:
            img_path = img_dir / f"{img_id_val}{img_ext}"
            actual_id = f"{img_id_val}{img_ext}"
            
        if img_path.exists():
            row_dict = row.to_dict()
            row_dict['_actual_img_path'] = str(img_path)
            row_dict['_actual_id'] = actual_id
            valid_rows.append(row_dict)
            
    df = pd.DataFrame(valid_rows)
    
    # Map conditions to unified categories
    if 'disability' in df.columns:
        df['unified_condition'] = df['disability'].apply(map_condition)
    elif 'condition' in df.columns:
        df['unified_condition'] = df['condition'].apply(map_condition)
    else:
        df['unified_condition'] = "Neutral"

    # Filter out anything that couldn't be mapped
    df = df[df['unified_condition'] != "Other"]

    conditions = ["Neutral", "Mobility", "Vision", "Hearing", "Mental Health"]
    samples_per_cond = n_samples // len(conditions)
    remainder = n_samples % len(conditions)

    sampled_dfs = []
    
    for i, cond in enumerate(conditions):
        cond_df = df[df['unified_condition'] == cond]
        
        # Distribute remainder
        n_to_sample = samples_per_cond + (1 if i < remainder else 0)
        
        if len(cond_df) >= n_to_sample:
            sampled_dfs.append(cond_df.sample(n=n_to_sample, random_state=42))
        else:
            # If not enough, take all
            sampled_dfs.append(cond_df)
            print(f"Warning: Not enough {cond} images in {model_name}. Needed {n_to_sample}, found {len(cond_df)}.")

    final_sample = pd.concat(sampled_dfs)
    
    # In case we didn't get enough (due to shortage in some conditions), sample from the remainder
    if len(final_sample) < n_samples:
        remaining_needed = n_samples - len(final_sample)
        remaining_pool = df[~df.index.isin(final_sample.index)]
        if len(remaining_pool) >= remaining_needed:
            fill_sample = remaining_pool.sample(n=remaining_needed, random_state=42)
            final_sample = pd.concat([final_sample, fill_sample])

    # Format the output dataframe
    manifest_rows = []
    for _, row in final_sample.iterrows():
        manifest_row = {
            'image_id': row['_actual_id'],
            'model': model_name,
            'image_path': row['_actual_img_path'],
            'profession': str(row.get('profession', 'N/A')),
            'age': str(row.get('age_bin', 'N/A')),
            'disability_condition': row['unified_condition'],
            **{attr: str(row.get(attr, 'N/A')) for attr in ATTRIBUTES}
        }
        manifest_rows.append(manifest_row)
        
    return pd.DataFrame(manifest_rows)


def main():
    print("Loading datasets...")
    flux_df = pd.read_csv(FLUX_CSV)
    sd35_df = pd.read_csv(SD35_CSV)
    nano_df = pd.read_csv(NANO_CSV)
    
    print("Sampling datasets...")
    # Using specific ID columns and extensions for each pipeline
    flux_sample = sample_dataset(flux_df, 50, 'dcdd_id', 'FLUX.1 Dev', FLUX_IMG_DIR, '.jpeg')
    sd35_sample = sample_dataset(sd35_df, 50, 'dcdd_id', 'SD 3.5 Large', SD35_IMG_DIR, '.jpeg')
    nano_sample = sample_dataset(nano_df, 100, 'custom_id', 'Nano Banana', NANO_IMG_DIR, '.png')
    
    final_manifest = pd.concat([flux_sample, sd35_sample, nano_sample], ignore_index=True)
    
    # Shuffle the final manifest to prevent sequential annotation bias
    final_manifest = final_manifest.sample(frac=1, random_state=42).reset_index(drop=True)
    
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    final_manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path} ({len(final_manifest)} rows)")
    
    # Generate sampling report
    report_path = OUTPUT_DIR / "sampling_report.md"
    with open(report_path, "w") as f:
        f.write("# Sampling Report\n\n")
        f.write("A reproducible validation sample was generated utilizing `random_state=42`.\n\n")
        f.write("## Strata Distribution\n")
        f.write(final_manifest.groupby(['model', 'disability_condition']).size().to_frame("count").reset_index().to_string(index=False))
        f.write("\n\n## Total Images per Model\n")
        f.write(final_manifest['model'].value_counts().to_frame("count").reset_index().to_string(index=False))
        f.write("\n\n*Note: The manifest was subsequently shuffled to prevent sequential annotation bias based on model or condition.*")
        
    print("Saved sampling report.")

if __name__ == "__main__":
    main()
