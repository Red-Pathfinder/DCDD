import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(r"C:\Study\Coding\Research\DCDD\Drift")
WORKING_DIR = BASE_DIR / "human_validation_reconstruction"
MANIFEST_PATH = WORKING_DIR / "sample_manifest.csv"
ANNOTATIONS_PATH = WORKING_DIR / "human_annotations.csv"
REPORT_PATH = WORKING_DIR / "verification_report.md"

ATTRIBUTES = [
    "gender_presentation", "skin_tone", "sentiment",
    "physical_framing", "activity_engagement", "social_integration"
]

def main():
    report_lines = ["# Validation Sample & Annotation Verification Report\n"]
    
    if not MANIFEST_PATH.exists():
        report_lines.append("❌ `sample_manifest.csv` not found.\n")
        with open(REPORT_PATH, 'w', encoding='utf-8') as f: f.writelines(report_lines)
        return
        
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    report_lines.append("## 1. Sample Uniqueness & Completeness\n")
    total_samples = len(manifest_df)
    unique_samples = manifest_df['image_id'].nunique()
    
    if total_samples == unique_samples:
        report_lines.append(f"✅ Exact match: {total_samples} samples generated, 0 duplicate image IDs.\n")
    else:
        report_lines.append(f"❌ ERROR: {total_samples} samples but only {unique_samples} unique IDs.\n")

    # Target model counts
    flux_n = len(manifest_df[manifest_df['model'] == 'FLUX.1 Dev'])
    sd35_n = len(manifest_df[manifest_df['model'] == 'SD 3.5 Large'])
    nano_n = len(manifest_df[manifest_df['model'] == 'Nano Banana'])
    
    report_lines.append(f"- FLUX images: {flux_n} (Expected: 50)\n")
    report_lines.append(f"- SD 3.5 images: {sd35_n} (Expected: 50)\n")
    report_lines.append(f"- Nano Banana images: {nano_n} (Expected: 100)\n\n")

    report_lines.append("## 2. Image File Existence\n")
    missing_images = []
    for _, row in manifest_df.iterrows():
        img_path = Path(str(row['image_path']))
        if not img_path.exists():
            missing_images.append(str(img_path))
            
    if len(missing_images) == 0:
        report_lines.append("✅ All 200 sampled image files exist on disk.\n\n")
    else:
        report_lines.append(f"❌ ERROR: {len(missing_images)} image files are missing!\n")
        for p in missing_images[:10]:
            report_lines.append(f"  - Missing: {p}\n")
        report_lines.append("\n")

    report_lines.append("## 3. VLM Label Completeness\n")
    missing_vlm = 0
    for attr in ATTRIBUTES:
        missing_vlm += manifest_df[attr].isna().sum()
    
    if missing_vlm == 0:
        report_lines.append("✅ All sampled images have complete VLM labels for the 6 attributes.\n\n")
    else:
        report_lines.append(f"❌ ERROR: {missing_vlm} missing VLM labels detected in manifest.\n\n")


    report_lines.append("## 4. Annotation Completeness\n")
    if not ANNOTATIONS_PATH.exists():
        report_lines.append("⚠️ `human_annotations.csv` not found (Annotation not yet started).\n")
    else:
        try:
            ann_df = pd.read_csv(ANNOTATIONS_PATH)
            if len(ann_df) == 0:
                report_lines.append("⚠️ `human_annotations.csv` is empty (Annotation not yet started).\n")
            else:
                expected_annotations = total_samples * len(ATTRIBUTES)
                actual_annotations = len(ann_df)
                
                if actual_annotations == expected_annotations:
                    report_lines.append(f"✅ Exact match: {actual_annotations} annotations complete (6 per image).\n")
                else:
                    report_lines.append(f"⚠️ Incomplete: {actual_annotations} out of {expected_annotations} expected annotations complete.\n")
                    
                # Check mapping
                invalid_ids = ann_df[~ann_df['image_id'].astype(str).isin(manifest_df['image_id'].astype(str))]
                if len(invalid_ids) == 0:
                    report_lines.append("✅ All annotations map to valid sampled images.\n")
                else:
                    report_lines.append(f"❌ ERROR: {len(invalid_ids)} annotations map to unknown image IDs!\n")
        except pd.errors.EmptyDataError:
            report_lines.append("⚠️ `human_annotations.csv` is empty (Annotation not yet started).\n")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"Verification report saved to {REPORT_PATH.name}")

if __name__ == "__main__":
    main()
