import os
import pandas as pd
from pathlib import Path
from PIL import Image

# ─── PATHS ──────────────────────────────────────────────────────────────────────
CSV_PATH = Path(r"C:\Study\Coding\Research\Drift\data_processed\03_nanobanana_attributes\nanobanana_attributes_final.csv")
IMAGE_DIR = Path(r"C:\Study\Coding\Research\Drift\data_raw\nanobanana_2_pipeline")

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
ATTRIBUTES = [
    "image_style", "gender_presentation", "perceived_race", "skin_tone",
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement",
    "social_integration"
]

# Words that trigger the "Ambiguous" flag
AMBIGUOUS_KEYWORDS = ["ambiguous", "unclear", "indeterminate"]

def is_ambiguous(val):
    """Checks if a cell contains any of the ambiguous keywords."""
    if pd.isna(val): return False
    text = str(val).lower()
    return any(keyword in text for keyword in AMBIGUOUS_KEYWORDS)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("[*] Loading Nanobanana data...")
    df = pd.read_csv(CSV_PATH)
    
    # Filter for rows that have at least one ambiguous attribute
    # We explicitly exclude "Safety Refusal" because those don't have images to view
    ambiguous_rows = []
    for _, row in df.iterrows():
        # Skip if it was a complete API safety refusal
        if row.get('image_style') == 'Safety Refusal':
            continue
            
        # Check if any attribute is ambiguous
        is_row_ambiguous = any(is_ambiguous(row[attr]) for attr in ATTRIBUTES)
        if is_row_ambiguous:
            ambiguous_rows.append(row)
            
    total_ambiguous = len(ambiguous_rows)
    print(f"[*] Found {total_ambiguous} images flagged as ambiguous by the VLM.\n")
    input("Press ENTER to start the manual review process...")
    
    for idx, row in enumerate(ambiguous_rows, start=1):
        clear_terminal()
        custom_id = row['custom_id']
        img_path = IMAGE_DIR / f"{custom_id}.png"
        
        print("=" * 70)
        print(f" IMAGE REVIEW: {idx} of {total_ambiguous}")
        print("=" * 70)
        print(f" ID:           {custom_id}")
        print(f" Profession:   {row.get('profession', 'N/A').title()}")
        print(f" Condition:    {row.get('condition', 'N/A').title()}")
        print(f" Temperature:  {row.get('temperature', 'N/A')}")
        print("-" * 70)
        print(" VLM EXTRACTED ATTRIBUTES:")
        print("-" * 70)
        
        for attr in ATTRIBUTES:
            val = str(row[attr])
            # Highlight the specific attributes that failed the VLM check
            if is_ambiguous(val):
                print(f"  [AMBIGUOUS]  {attr.replace('_', ' ').title().ljust(20)} : {val}")
            else:
                print(f"               {attr.replace('_', ' ').title().ljust(20)} : {val}")
        
        print("-" * 70)
        
        # Open the image using the default Windows Photo Viewer
        if img_path.exists():
            try:
                img = Image.open(img_path)
                img.show()
            except Exception as e:
                print(f"[!] Could not open image: {e}")
        else:
            print(f"[!] ERROR: Image file not found at {img_path}")
            
        print("\nLook at the image that just opened.")
        print("Is the image actually distorted? Or did the VLM just fail to understand it?")
        user_input = input("\nPress ENTER for next image, or type 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("\n[*] Exiting human validation. Good luck on the call with Srikant Sir!")
            break

if __name__ == "__main__":
    main()