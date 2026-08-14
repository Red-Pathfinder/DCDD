"""
Purpose:
Interactive manual review queue to correct VLM ambiguous labels for Nano Banana 2.

Paper Sections:
Section 3.4

Input Files:
05_data/nanobanana_attributes_final.csv, 05_data/human_overrides.csv (optional)

Output Files:
05_data/human_overrides.csv

Dependencies:
pandas, PIL

Approximate Runtime:
Interactive (depends on user)
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import csv

# Dynamic relative path to supplementary root
BASE_DIR = Path(__file__).resolve().parent.parent


# ─── PATHS FOR GEMINI (NANOBANANA) ──────────────────────────────────────────────
CSV_PATH = BASE_DIR / "05_data" / "nanobanana_attributes_final.csv"
IMAGE_DIR = BASE_DIR / "05_data" / "nanobanana_2_pipeline"
OVERRIDES_CSV = BASE_DIR / "05_data" / "human_overrides.csv"

# ─── CONFIGURATION & VALID CATEGORIES ───────────────────────────────────────────
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

AMBIGUOUS_KEYWORDS = ["ambiguous", "unclear", "indeterminate"]

def is_ambiguous(val):
    if pd.isna(val): return False
    return any(keyword in str(val).lower() for keyword in AMBIGUOUS_KEYWORDS)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_existing_overrides():
    """Loads already reviewed IDs so we don't ask you twice."""
    if not OVERRIDES_CSV.exists():
        return set()
    df_overrides = pd.read_csv(OVERRIDES_CSV)
    return set(zip(df_overrides['master_id'], df_overrides['attribute_name']))

def save_override(master_id, attribute_name, corrected_value):
    """Saves to CSV instantly so you never lose progress."""
    # Ensure the directory exists
    OVERRIDES_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = OVERRIDES_CSV.exists()
    with open(OVERRIDES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['master_id', 'attribute_name', 'corrected_value'])
        writer.writerow([master_id, attribute_name, corrected_value])

def main():
    print("[*] Loading Gemini data and previous progress...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find the Gemini CSV at {CSV_PATH}")
        return

    completed_reviews = load_existing_overrides()
    
    # Find all attributes that need human eyes
    tasks = []
    for _, row in df.iterrows():
        # Skip outright safety refusals (no image exists)
        if str(row.get('image_style')) == 'Safety Refusal':
            continue
            
        master_id = str(row['custom_id'])
        
        for attr in ATTRIBUTE_OPTIONS.keys():
            val = str(row[attr])
            if is_ambiguous(val):
                # Only add if we haven't already reviewed this specific attribute for this image
                if (master_id, attr) not in completed_reviews:
                    tasks.append({
                        'master_id': master_id,
                        'profession': row.get('profession', 'N/A'),
                        'condition': row.get('condition', 'N/A'),
                        'temperature': row.get('temperature', 'N/A'),
                        'attribute': attr,
                        'vlm_label': val
                    })
                    
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("\n🎉 ALL DONE! No ambiguous Gemini images left to review.")
        return
        
    print(f"[*] Found {total_tasks} specific attributes requiring human validation.\n")
    input("Press ENTER to start the manual review process...")
    
    for idx, task in enumerate(tasks, start=1):
        clear_terminal()
        master_id = task['master_id']
        img_path = IMAGE_DIR / f"{master_id}.png"
        
        # Open image automatically using Windows default viewer
        if img_path.exists():
            try:
                img = Image.open(img_path)
                img.show()
            except Exception:
                print(f"[!] Warning: Could not automatically open image {img_path.name}")
        else:
            print(f"[!] ERROR: Image file missing at {img_path}")
                
        print("=" * 75)
        print(f" GEMINI REVIEW {idx} of {total_tasks} | ID: {master_id}")
        print("=" * 75)
        print(f" Prompt Details : {task['profession'].title()} | {task['condition'].title()} | Temp: {task['temperature']}")
        print("-" * 75)
        print(f" ⚠️ The VLM flagged [{task['attribute'].replace('_', ' ').title()}] as: '{task['vlm_label']}'\n")
        
        options = ATTRIBUTE_OPTIONS[task['attribute']]
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
            
        print("\n  [s] Skip for now (Leave as ambiguous)")
        print("  [q] Save & Quit")
        print("-" * 75)
        
        while True:
            choice = input("Select correct label (1/2/3) or s/q: ").strip().lower()
            
            if choice == 'q':
                print(f"\n[*] Progress saved to {OVERRIDES_CSV.name}. Exiting...")
                return
            elif choice == 's':
                print("Skipped.")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(options):
                corrected_val = options[int(choice) - 1]
                save_override(master_id, task['attribute'], corrected_val)
                print(f"✓ Saved as: {corrected_val}")
                break
            else:
                print("Invalid input. Please select a valid number or 's' or 'q'.")

if __name__ == "__main__":
    main()