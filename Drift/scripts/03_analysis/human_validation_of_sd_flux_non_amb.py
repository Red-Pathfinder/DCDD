import os
import pandas as pd
from pathlib import Path
import csv
import subprocess

# ─── CORE SYSTEM PATHS ────────────────────────────────────────────────────────
FLUX_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2\attributes_v2.csv")
SD35_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\02_sd35_attributes_v2\attributes_v2.csv")

FLUX_IMAGE_DIR = Path(r"C:\Study\Coding\Research\Drift\data_raw\flux_pipeline")
SD35_IMAGE_DIR = Path(r"C:\Study\Coding\Research\Drift\data_raw\sd35_pipeline") 

HUMAN_OVERRIDES_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\human_overrides.csv")

ATTRIBUTE_OPTIONS = {
    "image_style": ["Photorealistic", "Stylized/synthetic", "Unclear / mixed style"],
    "gender_presentation": ["Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation"],
    "perceived_race": ["White presenting", "Non-white presenting", "Ambiguous / indeterminate"],
    "skin_tone": ["Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate"],
    "gaze_direction": ["Engaged", "Disengaged", "Unclear / indeterminate"],
    "physical_framing": ["Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate"],
    "activity_engagement": ["Active engagement", "Passive/dependent engagement", "Unclear / indeterminate"],
    "social_integration": ["Socially integrated", "Socially isolated", "Unclear"]
}

AMBIGUOUS_KEYWORDS = ["ambiguous", "unclear", "indeterminate"]

def is_ambiguous(val, attr_name):
    """Returns True if the VLM was confused."""
    if pd.isna(val): return True
    val_lower = str(val).lower()
    
    # Do not manually resolve safety refusals
    if "safety refusal" in val_lower: return False 
    
    # FIX: Do not manually resolve Sentiment 'Neutral/unclear' (Handled via Task 3 Math Merge)
    if attr_name == "sentiment" and val_lower == "neutral/unclear": return False
    
    return any(keyword in val_lower for keyword in AMBIGUOUS_KEYWORDS)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def auto_open_image_and_cleanup(img_path):
    try:
        subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception: pass
    try: subprocess.Popen(['explorer.exe', str(img_path)])
    except Exception:
        try: os.startfile(img_path)
        except Exception: pass

def save_human_override(master_id, attribute_name, corrected_value):
    file_exists = HUMAN_OVERRIDES_CSV.exists()
    HUMAN_OVERRIDES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_OVERRIDES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['master_id', 'attribute_name', 'corrected_value'])
        writer.writerow([master_id, attribute_name, corrected_value])

def get_already_resolved_ids():
    if not HUMAN_OVERRIDES_CSV.exists(): return set()
    df_overrides = pd.read_csv(HUMAN_OVERRIDES_CSV)
    return set(zip(df_overrides['master_id'].astype(str), df_overrides['attribute_name'].astype(str)))

def scan_dataset(csv_path, img_dir, model_name, resolved_set):
    tasks = []
    if not csv_path.exists():
        print(f"[!] Warning: Could not find {csv_path.name}")
        return tasks
        
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        master_id = str(row.get('dcdd_id', row.get('custom_id', row.name)))
        csv_filename = str(row['filename']) if 'filename' in row else f"{master_id}.jpeg"
        img_path = img_dir / csv_filename
        
        if img_path.exists():
            # Only loop through non-sentiment attributes for manual resolution
            for attr in ATTRIBUTE_OPTIONS.keys():
                if attr in df.columns:
                    vlm_label = str(row[attr])
                    if is_ambiguous(vlm_label, attr):
                        if (master_id, attr) not in resolved_set:
                            tasks.append({
                                'model': model_name,
                                'master_id': master_id,
                                'img_path': img_path,
                                'attribute': attr,
                                'vlm_label': vlm_label
                            })
    return tasks

def main():
    clear_terminal()
    print("[*] Initializing Ambiguity Resolution Engine (Task 2)...")
    
    resolved_set = get_already_resolved_ids()
    print(f"[*] Found {len(resolved_set)} previously resolved overrides.")
    
    tasks = []
    tasks.extend(scan_dataset(FLUX_CSV, FLUX_IMAGE_DIR, "FLUX.1 Dev", resolved_set))
    tasks.extend(scan_dataset(SD35_CSV, SD35_IMAGE_DIR, "SD 3.5 Large", resolved_set))
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("\n🎉 ALL AMBIGUOUS CASES RESOLVED! Task 2 is complete.")
        return
        
    print(f"\n[*] Found {total_tasks} remaining genuine ambiguous predictions.")
    input("Press ENTER to begin resolving...")

    for idx, task in enumerate(tasks, start=1):
        clear_terminal()
        auto_open_image_and_cleanup(task['img_path'])
        
        print("=" * 80)
        print(f" 🚨 AMBIGUITY RESOLVER [{idx}/{total_tasks}] | Model: {task['model']}")
        print("=" * 80)
        print(f" ID: {task['master_id']}")
        print(f" 📋 TARGET COLUMN : {task['attribute'].replace('_', ' ').upper()}")
        print(f" 🤖 VLM PREDICTION: {task['vlm_label']}")
        print("-" * 80)
        print(f" Select the CORRECT human label for [{task['attribute'].upper()}]:")
        
        options = ATTRIBUTE_OPTIONS[task['attribute']]
        for i, opt in enumerate(options, start=1):
            print(f"  [{i}] {opt}")
        print("  [s] Skip for now")
        print("  [q] Quit and save progress")
        print("-" * 80)
        
        while True:
            choice = input(f" Input selection: ").strip().lower()
            if choice == 'q':
                try: subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: pass
                print(f"\n[*] Progress saved safely.")
                return
            elif choice == 's':
                print(" [*] Skipped.")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(options):
                corrected_val = options[int(choice) - 1]
                save_human_override(task['master_id'], task['attribute'], corrected_val)
                print(f" ✓ Saved correction: {corrected_val}")
                break
            else:
                print(" [!] Invalid selection.")

    try: subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass
    
    clear_terminal()
    print("🎉 ALL AMBIGUOUS CASES RESOLVED! Task 2 is officially complete.")

if __name__ == "__main__":
    main()