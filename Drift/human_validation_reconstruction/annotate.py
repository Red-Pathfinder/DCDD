import os
import csv
import pandas as pd
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"C:\Study\Coding\Research\DCDD\Drift")
WORKING_DIR = BASE_DIR / "human_validation_reconstruction"
MANIFEST_PATH = WORKING_DIR / "sample_manifest.csv"
ANNOTATIONS_PATH = WORKING_DIR / "human_annotations.csv"

# The 6 specific attributes and their allowed categories
ATTRIBUTES = {
    "gender_presentation": ["Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation"],
    "skin_tone": ["Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate"],
    "sentiment": ["Positive", "Negative", "Neutral/unclear"],
    "physical_framing": ["Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate"],
    "activity_engagement": ["Active engagement", "Passive/dependent engagement", "Unclear / indeterminate"],
    "social_integration": ["Socially integrated", "Socially isolated", "Unclear"]
}

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_completed_annotations():
    """Returns a set of (image_id, attribute) tuples that have already been annotated."""
    if not ANNOTATIONS_PATH.exists():
        return set()
    try:
        df = pd.read_csv(ANNOTATIONS_PATH)
        return set(zip(df['image_id'].astype(str), df['attribute'].astype(str)))
    except Exception:
        return set()

def save_annotation(image_id, model, attribute, human_label):
    """Saves a single annotation immediately."""
    file_exists = ANNOTATIONS_PATH.exists()
    with open(ANNOTATIONS_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image_id', 'model', 'attribute', 'human_label', 'timestamp'])
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        writer.writerow([image_id, model, attribute, human_label, timestamp])

def main():
    if not MANIFEST_PATH.exists():
        print("[!] ERROR: sample_manifest.csv not found.")
        return

    df_manifest = pd.read_csv(MANIFEST_PATH)
    completed = get_completed_annotations()
    
    tasks = []
    # Identify what needs annotation
    for _, row in df_manifest.iterrows():
        img_id = str(row['image_id'])
        model = str(row['model'])
        img_path = Path(str(row['image_path']))
        
        for attr in ATTRIBUTES.keys():
            if (img_id, attr) not in completed:
                tasks.append({
                    'image_id': img_id,
                    'model': model,
                    'image_path': img_path,
                    'attribute': attr,
                    'vlm_label': str(row.get(attr, 'N/A'))
                })

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("🎉 All images have been fully annotated! No tasks remaining.")
        return

    print(f"[*] Found {total_tasks} remaining attribute annotations across {len(df_manifest)} images.")
    input("Press ENTER to start annotation (BLIND MODE - VLM labels are hidden)...")
    
    current_image_id = None
    img_viewer = None
    
    for idx, task in enumerate(tasks, start=1):
        clear_terminal()
        img_id = task['image_id']
        img_path = task['image_path']
        
        # If we switched images, show the new one
        if current_image_id != img_id:
            try:
                img = Image.open(str(img_path))
                img.show()
            except Exception as e1:
                try:
                    os.startfile(str(img_path))
                except Exception as e2:
                    print(f"[!] Warning: Could not auto-open image {img_path.name}")
            current_image_id = img_id

        print("=" * 80)
        print(f" 🕵️ BLIND ANNOTATION [{idx}/{total_tasks}]")
        print("=" * 80)
        print(f" Image ID: {img_id}")
        print(f" Model: {task['model']}")
        print(f"\n 📋 CURRENT ATTRIBUTE: {task['attribute'].replace('_', ' ').upper()}")
        print("-" * 80)
        
        options = ATTRIBUTES[task['attribute']]
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        print("\n  [q] Save & Quit (resume later)")
        print("-" * 80)
        
        while True:
            choice = input(" Select the correct category (1/2/3) or 'q': ").strip().lower()
            
            if choice == 'q':
                print(f"\n[*] Progress safely saved to {ANNOTATIONS_PATH.name}. Exiting.")
                return
            elif choice.isdigit() and 1 <= int(choice) <= len(options):
                human_val = options[int(choice) - 1]
                save_annotation(
                    image_id=img_id,
                    model=task['model'],
                    attribute=task['attribute'],
                    human_label=human_val
                )
                print(f" ✓ Saved: {human_val}")
                break
            else:
                print(" [!] Invalid input. Choose 1, 2, 3 or q.")

    clear_terminal()
    print("🎉 ALL ANNOTATIONS COMPLETE!")

if __name__ == "__main__":
    main()
