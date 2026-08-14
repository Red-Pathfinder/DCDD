import os
import glob
import re
import shutil
import uuid
import random
import csv
from collections import defaultdict

def main():
    base_dir = "/Users/red-pathfinder/Material Girl/Coding/DCDD/Drift"
    data_raw_dir = os.path.join(base_dir, "data_raw")
    study_dir = os.path.join(base_dir, "human_annotation_study")
    
    blinded_dir = os.path.join(study_dir, "blinded_images")
    private_dir = os.path.join(study_dir, "private")
    app_dir = os.path.join(study_dir, "annotation_app")
    
    os.makedirs(blinded_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)
    
    # We need exactly 20 complete cells per model.
    # A complete cell has: neutral, mobility, vision, hearing, mental health
    models = {
        "flux": "flux_pipeline",
        "sd35": "sd35_pipeline",
        "nanobanana": "nanobanana_2_pipeline"
    }
    
    conditions_required = {
        "neutral", 
        "mobility_impairment", 
        "vision_impairment", 
        "hearing_impairment", 
        "mental_health_condition"
    }

    all_valid_cells = {
        "flux": [],
        "sd35": [],
        "nanobanana": []
    }
    
    print("Scanning images...")
    
    # 1. Parse flux and sd35
    for model in ["flux", "sd35"]:
        folder = os.path.join(data_raw_dir, models[model])
        files = glob.glob(os.path.join(folder, "*.jpeg"))
        
        cells = defaultdict(dict)
        for f in files:
            basename = os.path.basename(f)
            condition = None
            for c in conditions_required:
                if basename.endswith(f"{c}.jpeg"):
                    condition = c
                    break
            
            if not condition:
                continue
                
            cell_key = basename.replace(f"_{condition}.jpeg", "")
            cells[cell_key][condition] = f
            
        for cell_key, cond_dict in cells.items():
            if set(cond_dict.keys()) == conditions_required:
                parts = cell_key.split("_")
                pair_id = parts[0]
                
                age = "unknown"
                if cell_key.endswith("young_adult_18_25"):
                    age = "young_adult_18_25"
                elif cell_key.endswith("adult_26_40"):
                    age = "adult_26_40"
                elif cell_key.endswith("senior_60"):
                    age = "senior_60"
                
                profession = cell_key[len(pair_id)+1 : -len(age)-1]
                
                all_valid_cells[model].append({
                    "cell_id": cell_key,
                    "pair_id": pair_id,
                    "profession": profession,
                    "age": age,
                    "files": cond_dict
                })
                
    # 2. Parse Nano Banana
    nb_folder = os.path.join(data_raw_dir, models["nanobanana"])
    nb_files = glob.glob(os.path.join(nb_folder, "*.png"))
    
    nb_cells = defaultdict(dict)
    for f in nb_files:
        basename = os.path.basename(f)
        
        condition = None
        if basename.startswith("neutral_"):
            condition = "neutral"
        else:
            for c in conditions_required:
                if c != "neutral" and c in basename:
                    condition = c
                    break
                    
        if not condition:
            continue
            
        temp_match = re.search(r'(_t\d\.\d+)\.png$', basename)
        if not temp_match:
            continue
        temp = temp_match.group(1)
        
        middle = basename
        if condition == "neutral":
            middle = middle.replace("neutral_", "")
            middle = middle.replace(f"{temp}.png", "")
        else:
            middle = middle.replace("cond_", "")
            middle = middle.replace(f"_{condition}", "")
            middle = middle.replace(f"{temp}.png", "")
            
        age = "unknown"
        if middle.endswith("young_adult_18_25"):
            age = "young_adult_18_25"
        elif middle.endswith("adult_26_40"):
            age = "adult_26_40"
        elif middle.endswith("senior_60"):
            age = "senior_60"
            
        profession = middle[:-len(age)-1] if middle.endswith(age) else middle
        
        cell_key = f"{profession}_{age}{temp}"
        nb_cells[cell_key][condition] = f
        
    for cell_key, cond_dict in nb_cells.items():
        if set(cond_dict.keys()) == conditions_required:
            temp = cell_key[-4:]
            prof_age = cell_key[:-4]
            age = "unknown"
            if prof_age.endswith("young_adult_18_25"):
                age = "young_adult_18_25"
            elif prof_age.endswith("adult_26_40"):
                age = "adult_26_40"
            elif prof_age.endswith("senior_60"):
                age = "senior_60"
            profession = prof_age[:-len(age)-1]
            
            all_valid_cells["nanobanana"].append({
                "cell_id": cell_key,
                "pair_id": temp,
                "profession": profession,
                "age": age,
                "files": cond_dict
            })
            
    for m in models:
        print(f"{m} valid complete cells: {len(all_valid_cells[m])}")
        
    for m in models:
        if len(all_valid_cells[m]) < 20:
            print(f"ERROR: Not enough complete cells for model {m}. Found {len(all_valid_cells[m])}.")
            return
            
    # 3. Stratified sampling
    random.seed(42)
    selected_cells = []
    
    for m in models:
        sorted_cells = sorted(all_valid_cells[m], key=lambda x: x['cell_id'])
        sampled = random.sample(sorted_cells, 20)
        for s in sampled:
            s['model'] = m
        selected_cells.extend(sampled)
        
    assert len(selected_cells) == 60, f"Expected 60 cells, got {len(selected_cells)}"
    
    # 4. Process and anonymize
    manifest_rows = []
    image_list_for_app = []
    
    for cell in selected_cells:
        for condition in conditions_required:
            original_path = cell["files"][condition]
            original_filename = os.path.basename(original_path)
            
            anon_id = f"img_{uuid.uuid4().hex[:8]}"
            ext = os.path.splitext(original_filename)[1]
            anon_filename = f"{anon_id}{ext}"
            
            dest_path = os.path.join(blinded_dir, anon_filename)
            shutil.copy2(original_path, dest_path)
            
            manifest_rows.append({
                "anonymized_image_id": anon_id,
                "anonymized_filename": anon_filename,
                "original_filename": original_filename,
                "model": cell["model"],
                "profession": cell["profession"],
                "age": cell["age"],
                "condition": condition,
                "pair_id": cell["cell_id"]
            })
            
            image_list_for_app.append(anon_filename)
            
    assert len(manifest_rows) == 300, f"Expected 300 images, got {len(manifest_rows)}"
    
    # 5. Save manifest
    manifest_path = os.path.join(private_dir, "image_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "anonymized_image_id", "anonymized_filename", "original_filename", 
            "model", "profession", "age", "condition", "pair_id"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Save image list for JS
    random.shuffle(image_list_for_app)
    with open(os.path.join(app_dir, "image_list.js"), "w") as f:
        f.write(f"const imageList = {image_list_for_app};\n")
        
    # Generate sampling log
    with open(os.path.join(study_dir, "sampling", "sampling_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "anonymized_image_id", "model", "profession", "age", "condition", "original_filename", "pair_id"
        ])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({
                "anonymized_image_id": row["anonymized_image_id"],
                "model": row["model"],
                "profession": row["profession"],
                "age": row["age"],
                "condition": row["condition"],
                "original_filename": row["original_filename"],
                "pair_id": row["pair_id"]
            })
            
    print("SUCCESS: Sampled 300 images. Manifest written. Image list written.")
    
if __name__ == "__main__":
    main()
