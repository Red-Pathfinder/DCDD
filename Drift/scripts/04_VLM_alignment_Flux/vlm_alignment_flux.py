import os
import pandas as pd
from pathlib import Path
import csv
import random
import subprocess

# ─── CORE SYSTEM PATHS ────────────────────────────────────────────────────────
FLUX_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2\attributes_v2.csv")
FLUX_IMAGE_DIR = Path(r"C:\Study\Coding\Research\Drift\data_raw\flux_pipeline") 
AUDIT_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\vlm_alignment_audit.csv")
HUMAN_OVERRIDES_CSV = Path(r"C:\Study\Coding\Research\Drift\data_processed\human_overrides.csv")

# Codebooks mapped directly from our master configuration schema
ATTRIBUTE_OPTIONS = {
    "image_style": ["Photorealistic", "Stylized/synthetic", "Unclear / mixed style"],
    "gender_presentation": ["Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation"],
    "perceived_race": ["White presenting", "Non-white presenting", "Ambiguous / indeterminate"],
    "skin_tone": ["Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate"],
    "sentiment": ["Positive", "Negative", "Neutral/unclear"],
    "gaze_direction": ["Engaged", "Disengaged", "Unclear / indeterminate"],
    "physical_framing": ["Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate"],
    "activity_engagement": ["Active engagement", "Passive/dependent engagement", "Unclear / indeterminate"],
    "social_integration": ["Socially integrated", "Socially isolated", "Unclear"]
}

AMBIGUOUS_KEYWORDS = ["ambiguous", "unclear", "indeterminate", "safety refusal"]

def is_confident(val):
    if pd.isna(val): return False
    val_lower = str(val).lower()
    return not any(keyword in val_lower for keyword in AMBIGUOUS_KEYWORDS)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def auto_open_image_and_cleanup(img_path):
    try:
        subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        subprocess.Popen(['explorer.exe', str(img_path)])
    except Exception:
        try: os.startfile(img_path)
        except Exception: pass

def save_human_override(master_id, attribute_name, corrected_value):
    """Dynamically appends label fixes into your main human corrections ledger."""
    file_exists = HUMAN_OVERRIDES_CSV.exists()
    HUMAN_OVERRIDES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_OVERRIDES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['master_id', 'attribute_name', 'corrected_value'])
        writer.writerow([master_id, attribute_name, corrected_value])

def main():
    clear_terminal()
    print("[*] Initializing Ground-Truth Alignment + Interactive Correction Pipeline...")
    
    if not FLUX_CSV.exists() or not FLUX_IMAGE_DIR.exists():
        print("[!] SYSTEM PATH ERROR: Verify data indices and repository roots.")
        return

    df = pd.read_csv(FLUX_CSV)
    cond_col = 'disability' if 'disability' in df.columns else 'condition'
    
    df_vision = df[df[cond_col].astype(str).str.lower() == 'vision impairment']
    if df_vision.empty:
        print(f"[!] Zero experimental records matched 'vision impairment'.")
        return

    tasks = []
    for _, row in df_vision.iterrows():
        master_id = str(row.get('dcdd_id', row.get('custom_id', row.name)))
        csv_filename = str(row['filename']) if 'filename' in row else f"{master_id}.jpeg"
        img_path = FLUX_IMAGE_DIR / csv_filename
        
        if img_path.exists():
            for attr in ATTRIBUTE_OPTIONS.keys():
                if attr in df.columns:
                    vlm_label = str(row[attr])
                    if is_confident(vlm_label):
                        tasks.append({
                            'master_id': master_id,
                            'img_path': img_path,
                            'attribute': attr,
                            'vlm_label': vlm_label
                        })
            
    total_confident = len(tasks)
    if total_confident == 0:
        print("[!] No evaluation points available to audit.")
        return

    random.seed(42)
    random.shuffle(tasks)
    sample_size = min(50, total_confident)
    tasks_to_review = tasks[:sample_size]
    
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIT_CSV.exists():
        with open(AUDIT_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['master_id', 'attribute', 'vlm_label', 'human_agrees'])

    # Runtime Execution Loop
    matches = 0
    for idx, task in enumerate(tasks_to_review, start=1):
        clear_terminal()
        auto_open_image_and_cleanup(task['img_path'])
        
        print("=" * 80)
        print(f" 🔍 INTEGRITY AUDIT [{idx}/{sample_size}] | ID: {task['master_id']}")
        print("=" * 80)
        print(f" 📋 TARGET COLUMN : {task['attribute'].replace('_', ' ').upper()}")
        print(f" 🤖 VLM PREDICTION: {task['vlm_label']}")
        print("-" * 80)
        
        while True:
            choice = input(" Does human judgment align with the VLM prediction? (y/n) [or 'q' to quit]: ").strip().lower()
            if choice == 'q':
                try: subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except: pass
                print(f"\n[*] Run terminated safely.")
                return
            elif choice == 'y':
                matches += 1
                with open(AUDIT_CSV, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([task['master_id'], task['attribute'], task['vlm_label'], True])
                break
            elif choice == 'n':
                # Write disagreement to the audit log
                with open(AUDIT_CSV, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([task['master_id'], task['attribute'], task['vlm_label'], False])
                
                # INTERACTIVE CORRECTION SUB-MENU
                print("\n" + "-"*40)
                print(f" Select the CORRECT human label for [{task['attribute'].upper()}]:")
                options = ATTRIBUTE_OPTIONS[task['attribute']]
                for i, opt in enumerate(options, start=1):
                    print(f"  [{i}] {opt}")
                print("-" * 40)
                
                while True:
                    fix_choice = input(f" Input correct number (1-{len(options)}): ").strip()
                    if fix_choice.isdigit() and 1 <= int(fix_choice) <= len(options):
                        corrected_val = options[int(fix_choice) - 1]
                        save_human_override(task['master_id'], task['attribute'], corrected_val)
                        print(f" ✓ Saved correction: {corrected_val}")
                        break
                    else:
                        print(" [!] Invalid selection.")
                break
            else:
                print(" [!] Input error: Supply 'y', 'n', or 'q'.")

    # Final wrap summary
    try: subprocess.run(["taskkill", "/F", "/IM", "Microsoft.Photos.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass
    accuracy = (matches / sample_size) * 100
    clear_terminal()
    print(f"🏆 PIPELINE AUDIT RECORDED. Empirical Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()