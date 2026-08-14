import os
import pandas as pd
import json
import shutil
from pathlib import Path

# ─── EXACT PATHS ──────────────────────────────────────────────────────────────
RAW_DIR      = Path(r"C:\Study\Coding\Research\Drift\data_raw\flux_pipeline")
ATTR_DIR     = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes")
ARCHIVE_DIR  = Path(r"C:\Study\Coding\Research\Drift\archive_teens")

# Specific Files
META_CSV     = RAW_DIR / "metadata.csv"
CHECKPOINT   = RAW_DIR / "checkpoint.json"
ATTR_CSV     = ATTR_DIR / "attributes.csv"
JSONL_INPUT  = ATTR_DIR / "batch_input.jsonl"
JSONL_OUTPUT = ATTR_DIR / "batch_output.jsonl"

def deep_purge():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🚀 STARTING DEEP PURGE: RAW & PROCESSED FLUX DATA")
    print("="*60)

    # 1. PRUNE RAW METADATA & IMAGES
    if META_CSV.exists():
        df_meta = pd.read_csv(META_CSV)
        teens_meta = df_meta[df_meta['age_bin'] == 'teen (13-17)']
        clean_meta = df_meta[df_meta['age_bin'] != 'teen (13-17)']
        
        moved_count = 0
        for _, row in teens_meta.iterrows():
            src = RAW_DIR / row['filename']
            dst = ARCHIVE_DIR / row['filename']
            if src.exists():
                shutil.move(src, dst)
                moved_count += 1
        
        clean_meta.to_csv(META_CSV, index=False)
        print(f"[✓] {moved_count} raw images archived.")
        print(f"[✓] metadata.csv cleaned ({len(clean_meta)} rows).")

    # 2. PRUNE PROCESSED ATTRIBUTES CSV
    if ATTR_CSV.exists():
        df_attr = pd.read_csv(ATTR_CSV)
        clean_attr = df_attr[df_attr['age_bin'] != 'teen (13-17)']
        clean_attr.to_csv(ATTR_CSV, index=False)
        print(f"[✓] attributes.csv cleaned ({len(clean_attr)} rows).")

    # 3. PRUNE JSONL FILES (Input & Output)
    for jsonl_file in [JSONL_INPUT, JSONL_OUTPUT]:
        if jsonl_file.exists():
            clean_lines = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Check custom_id for 'teen'
                    if 'teen' not in data.get('custom_id', '').lower():
                        clean_lines.append(line)
            
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.writelines(clean_lines)
            print(f"[✓] {jsonl_file.name} scrubbed ({len(clean_lines)} entries).")

    # 4. SCRUB CHECKPOINT JSON
    if CHECKPOINT.exists():
        with open(CHECKPOINT, 'r') as f:
            completed_ids = json.load(f)
        clean_ids = [cid for cid in completed_ids if 'teen' not in cid.lower()]
        with open(CHECKPOINT, 'w') as f:
            json.dump(clean_ids, f, indent=4)
        print(f"[✓] checkpoint.json scrubbed.")

    print("\n" + "="*60)
    print("✨ DEEP PURGE COMPLETE. All evidence of 'teen' category removed.")
    print("="*60)

if __name__ == "__main__":
    deep_purge()