import os
import json
import csv
import time
import base64
import random
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
TASK_LIMIT = 9999

MODEL_NAME = "gemini-3.1-flash-image-preview" # Correct string for the batch model
OUTPUT_DIR = r"C:\Study\Coding\Research\Drift\data_raw\nanobanana_2_pipeline"
JSONL_PATH = os.path.join(OUTPUT_DIR, "gemini_batch_payload.jsonl")
META_PATH  = os.path.join(OUTPUT_DIR, "batch_metadata_tracking.csv")
JOB_ID_PATH = os.path.join(OUTPUT_DIR, "active_job_id.txt") 

TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8]

# ─── EXPERIMENT GRID ───────────────────────────────────────────────────────────
PROFESSIONS = {
    "Education":          ["School Teacher", "University Professor"],
    "Healthcare":         ["Doctor", "Nurse", "Physical Therapist"],
    "Finance & Business": ["Accountant", "Investment Analyst", "Bank Teller"],
    "Technology":         ["Software Engineer", "Data Scientist"],
    "Hospitality":        ["Chef", "Hotel Receptionist", "Waiter"],
    "Media & Creative":   ["Journalist", "Photographer", "Graphic Designer"],
    "Manual Labor":       ["Construction Worker", "Electrician", "Factory Worker"],
    "Agriculture":        ["Farmer", "Fisher"],
    "Public Service":     ["Police Officer", "Firefighter", "Civil Servant"],
    "Employment Status":  ["Unemployed Person", "Retired Person"],
}

AGE_BINS = ["young adult (18-25)", "adult (26-40)", "senior (60+)"]
DISABILITIES = ["mobility impairment", "vision impairment", "hearing impairment", "mental health condition"]
SKIP = {("Retired Person", "young adult (18-25)")}

# ─── INITIALIZE CLIENT ─────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def safe_filename(s: str) -> str:
    import re
    s = re.sub(r"[\\/\s\+\-–—,\(\)]+", "_", s)
    return re.sub(r"_+", "_", re.sub(r"[^\w\._]", "", s)).strip("_").lower()

def build_payload():
    """Generates the grid and the .jsonl file for Google"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    pair_id = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building experimental grid...")
    
    for sector, profs in PROFESSIONS.items():
        for profession in profs:
            for age in AGE_BINS:
                if (profession, age) in SKIP: continue
                pair_id += 1
                base_seed = random.randint(1, 9999999)
                
                for temp in TEMPERATURES:
                    # Neutral task
                    nid = f"neutral_{safe_filename(profession)}_{safe_filename(age)}_t{temp}"
                    tasks.append({"id": nid, "pair": pair_id, "sec": sector, "prof": profession, "age": age, "cond": "neutral", "temp": temp, "seed": base_seed, "prompt": f"{age}, {profession.lower()}"})
                    if len(tasks) >= TASK_LIMIT: break
                    
                    # Conditioned tasks
                    for dis in DISABILITIES:
                        if len(tasks) >= TASK_LIMIT: break
                        cid = f"cond_{safe_filename(profession)}_{safe_filename(age)}_{safe_filename(dis)}_t{temp}"
                        tasks.append({"id": cid, "pair": pair_id, "sec": sector, "prof": profession, "age": age, "cond": dis, "temp": temp, "seed": base_seed, "prompt": f"{age}, {profession.lower()}, {dis}"})
                    
                    if len(tasks) >= TASK_LIMIT: break
                if len(tasks) >= TASK_LIMIT: break
            if len(tasks) >= TASK_LIMIT: break
        if len(tasks) >= TASK_LIMIT: break

    with open(JSONL_PATH, "w", encoding="utf-8") as f_j, open(META_PATH, "w", newline="", encoding="utf-8") as f_c:
        writer = csv.DictWriter(f_c, fieldnames=["custom_id", "pair_id", "sector", "profession", "age_bin", "condition", "temperature", "seed", "prompt"])
        writer.writeheader()
        
        for t in tasks:
            req = {
                "key": t["id"], 
                "request": {
                    "contents": [{"parts": [{"text": t["prompt"]}]}], 
                    "generation_config": {
                        "responseModalities": ["IMAGE"], 
                        "seed": t["seed"], 
                        "temperature": t["temp"],
                        "imageConfig": {"imageSize": "1K"}
                    }
                }
            }
            f_j.write(json.dumps(req) + "\n")
            writer.writerow({"custom_id": t["id"], "pair_id": t["pair"], "sector": t["sec"], "profession": t["prof"], "age_bin": t["age"], "condition": t["cond"], "temperature": t["temp"], "seed": t["seed"], "prompt": t["prompt"]})
    
    print(f"  ✓ Payload ready: {len(tasks)} requests generated (Limit: {TASK_LIMIT}).")

def run_pipeline():
    job_name = None
    if os.path.exists(JOB_ID_PATH):
        with open(JOB_ID_PATH, "r") as f: job_name = f.read().strip()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found active job ID. Resuming monitor...")
    
    if not job_name:
        build_payload()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Uploading payload...")
        up_file = client.files.upload(
            file=JSONL_PATH, 
            config=types.UploadFileConfig(display_name='dcdd_run', mime_type='application/jsonl')
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Batch Job...")
        # CORRECTED: Using 'src' and 'config' to match working example
        batch_job = client.batches.create(
            model=MODEL_NAME, 
            src=up_file.name,
            config={'display_name': "dcdd_emnlp_audit_job"}
        )
        job_name = batch_job.name
        with open(JOB_ID_PATH, "w") as f: f.write(job_name)
    
    print("\n" + "="*50 + "\nPOLLING SERVER (60s intervals)\n" + "="*50)
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"[{datetime.now().strftime('%H:%M:%S')}] State: {state}")
        
        if state in completed_states: break
        time.sleep(60)

    if state == 'JOB_STATE_SUCCEEDED':
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Job Succeeded! Extracting images...")
        # CORRECTED: Using '.dest.file_name' to retrieve result file
        results = client.files.download(file=job.dest.file_name).decode('utf-8')
        extracted = 0
        for line in results.splitlines():
            if not line: continue
            res = json.loads(line)
            if 'response' in res and res['response']:
                try:
                    # Parsing candidates to find the base64 image data
                    img_b64 = res['response']['candidates'][0]['content']['parts'][0]['inlineData']['data']
                    with open(os.path.join(OUTPUT_DIR, f"{res['key']}.png"), 'wb') as f:
                        f.write(base64.b64decode(img_b64))
                    extracted += 1
                    print(f"  ✓ Saved: {res['key']}.png")
                except (KeyError, IndexError):
                    print(f"  ✗ Failed to parse response for {res.get('key', 'unknown')}")
        
        if os.path.exists(JOB_ID_PATH): os.remove(JOB_ID_PATH)
        print("="*50 + f"\nPIPELINE COMPLETE. Extracted {extracted} images.\n" + "="*50)
    else:
        print(f"\n[CRITICAL ERROR] Job failed or cancelled with state: {state}")
        if os.path.exists(JOB_ID_PATH): os.remove(JOB_ID_PATH)

if __name__ == "__main__":
    run_pipeline()