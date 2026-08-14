import os
import csv

def validate_study():
    base_dir = "/Users/red-pathfinder/Material Girl/Coding/DCDD/Drift/human_annotation_study"
    manifest_path = os.path.join(base_dir, "private", "image_manifest.csv")
    blinded_dir = os.path.join(base_dir, "blinded_images")
    
    if not os.path.exists(manifest_path):
        print("FAIL: Manifest file missing.")
        return False
        
    with open(manifest_path, "r") as f:
        reader = list(csv.DictReader(f))
        
    total_images = len(reader)
    if total_images != 300:
        print(f"FAIL: Expected 300 images, found {total_images}.")
        return False
        
    cells = {}
    models = set()
    anon_ids = set()
    original_names = set()
    
    for row in reader:
        cell_id = row['pair_id']
        if cell_id not in cells:
            cells[cell_id] = set()
        cells[cell_id].add(row['condition'])
        
        models.add(row['model'])
        
        if row['anonymized_image_id'] in anon_ids:
            print(f"FAIL: Duplicate anonymized ID {row['anonymized_image_id']}")
            return False
        anon_ids.add(row['anonymized_image_id'])
        
        if row['original_filename'] in original_names:
            print(f"FAIL: Duplicate original filename {row['original_filename']}")
            return False
        original_names.add(row['original_filename'])
        
        if not os.path.exists(os.path.join(blinded_dir, row['anonymized_filename'])):
            print(f"FAIL: Missing image file {row['anonymized_filename']}")
            return False
            
    if len(cells) != 60:
        print(f"FAIL: Expected exactly 60 counterfactual cells, found {len(cells)}")
        return False
        
    required_conditions = {"neutral", "mobility_impairment", "vision_impairment", "hearing_impairment", "mental_health_condition"}
    for cell_id, conditions in cells.items():
        if conditions != required_conditions:
            print(f"FAIL: Cell {cell_id} does not have all 5 conditions. It has {conditions}")
            return False
            
    if len(models) != 3:
        print(f"FAIL: Expected 3 models represented, found {len(models)}: {models}")
        return False
        
    # Check for metadata leakage in anonymized filenames
    for row in reader:
        filename = row['anonymized_filename'].lower()
        if any(x in filename for x in ['flux', 'sd35', 'nanobanana', 'adult', 'senior', 'neutral', 'vision', 'hearing', 'mobility', 'mental']):
            print(f"FAIL: Metadata leakage detected in anonymized filename: {filename}")
            return False
            
    print("========================================")
    print("      VALIDATION REPORT PASSED          ")
    print("========================================")
    print(f"Total Images: {total_images}")
    print(f"Total Cells: {len(cells)}")
    print("All cells contain exactly 5 conditions.")
    print("All 3 models represented.")
    print("No duplicate IDs or missing files.")
    print("No metadata leakage in filenames.")
    
    return True

if __name__ == "__main__":
    validate_study()
