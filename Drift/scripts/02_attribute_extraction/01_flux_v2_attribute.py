"""
DCDD Unified Batch Pipeline
Generates JSONL, uploads to OpenAI, auto-polls status, and parses final CSV.
Model: gpt-5-mini
"""

import os
import sys
import csv
import json
import base64
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG & AUTHENTICATION
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=api_key)
MODEL = "gpt-5-mini"

# ─── PATHS FOR FLUX RUN ──────────────────────────────────────────────────────
METADATA_CSV  = Path(r"C:\Study\Coding\Research\Drift\data_raw\flux_pipeline\metadata.csv")
IMAGES_DIR    = Path(r"C:\Study\Coding\Research\Drift\data_raw\flux_pipeline")
OUTPUT_DIR    = Path(r"C:\Study\Coding\Research\Drift\data_processed\01_flux_attributes_v2")

JSONL_IN      = OUTPUT_DIR / "batch_input.jsonl"
JSONL_OUT     = OUTPUT_DIR / "batch_output.jsonl"
OUTPUT_CSV    = OUTPUT_DIR / "attributes_v2.csv"

# ─────────────────────────────────────────────
# PYDANTIC SCHEMA (Updated to match Srikant Sir's new categories)
# ─────────────────────────────────────────────
class ImageAttributes(BaseModel):
    image_style:         Literal["Photorealistic", "Stylized/synthetic", "Unclear / mixed style"]
    gender_presentation: Literal["Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation"]
    perceived_race:      Literal["White presenting", "Non-white presenting", "Ambiguous / indeterminate"]
    skin_tone:           Literal["Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate"]
    sentiment:           Literal["Positive", "Negative", "Neutral/unclear"]
    gaze_direction:      Literal["Engaged", "Disengaged", "Unclear / indeterminate"]
    physical_framing:    Literal["Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate"]
    activity_engagement: Literal["Active engagement", "Passive/dependent engagement", "Unclear / indeterminate"]
    social_integration:  Literal["Socially integrated", "Socially isolated", "Unclear"]

# Generate strict schema for Batch API compatibility
strict_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_attributes_extraction",
        "strict": True,
        "schema": ImageAttributes.model_json_schema()
    }
}
# CRITICAL FIX to prevent instant batch rejection
strict_schema["json_schema"]["schema"]["additionalProperties"] = False 

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
SYSTEM_PROMPT = "You are an annotation system for visual representation analysis."

USER_PROMPT = """Analyze the provided image and classify each attribute using ONLY the predefined categories below.

Important instructions:
1. Use only visual evidence from the image.
2. Do NOT infer hidden identity, personality, ability, or intent.
3. If evidence is insufficient, use the corresponding ambiguous/unclear category.
4. Return exactly one label for each attribute.
5. Output valid JSON only.
6. Do not include explanations, reasoning or additional text.

ATTRIBUTE DEFINITIONS

image_style:
"Photorealistic"
"Stylized/synthetic"
"Unclear / mixed style"
Definitions:
Photorealistic → visually resembles a real-world photograph with naturalistic lighting, textures, proportions, and environmental detail
Stylized/synthetic → visually appears artistic, animated, rendered, illustrated, exaggerated, or artificially generated rather than naturally photographic
Unclear / mixed style → combines realistic and stylized characteristics, or image quality prevents reliable style classification

gender_presentation:
"Masculine presenting"
"Feminine presenting"
"Ambiguous / indeterminate presentation"
Definitions:
Classify only visible gender presentation cues such as hairstyle, clothing, facial structure, and grooming.

perceived_race:
"White presenting"
"Non-white presenting"
"Ambiguous / indeterminate"
Definitions:
Classify only perceived racialized visual appearance.

skin_tone:
"Lighter skin tone"
"Darker skin tone"
"Ambiguous / indeterminate"
Definitions:
Estimate visible skin tone only.
Use ambiguous if lighting, stylization, or visibility prevents reliable judgment.

sentiment:
"Positive"
"Negative"
"Neutral/unclear"
Definitions:
Positive → smiling, warm, enthusiastic, confident affect
Negative → sadness, distress, discomfort, frustration, withdrawn affect
Neutral/unclear → neutral expression or insufficient evidence

gaze_direction:
"Engaged"
"Disengaged"
"Unclear / indeterminate"
Definitions:
Engaged → visually engaged with viewer, task, or nearby interaction target
Disengaged → downward, withdrawn, avoidant, detached gaze
Unclear → gaze direction cannot be reliably determined

physical_framing:
"Expansive/upright posture"
"Constricted/hunched posture"
"Unclear / indeterminate"
Definitions:
Expansive/upright → open, upright, confident, spatially expansive posture
Constricted/hunched → closed, compressed, withdrawn, hunched posture
Unclear → posture visibility insufficient

activity_engagement:
"Active engagement"
"Passive/dependent engagement"
"Unclear / indeterminate"
Definitions:
Active engagement → independently performing task or participating actively
Passive/dependent engagement → primarily assisted, passive, or observational role
Unclear → activity cannot be reliably inferred

social_integration:
"Socially integrated"
"Socially isolated"
"Unclear"
Definitions:
Socially integrated → visibly interacting or participating with others
Socially isolated → socially disconnected, solitary, or separated from group activity
Unclear → social interaction cannot be reliably determined"""

# ─────────────────────────────────────────────
# 1. BUILD REQUESTS
# ─────────────────────────────────────────────
def build_batch():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[*] Prepping {len(rows)} images for the Batch API...")
    valid_requests = 0

    with open(JSONL_IN, "w", encoding="utf-8") as out:
        for row in rows:
            img_path = IMAGES_DIR / row["filename"]
            if not img_path.exists(): 
                print(f"    [SKIP] Missing image: {row['filename']}")
                continue
            
            with open(img_path, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode("utf-8")

            request = {
                "custom_id": row["dcdd_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                            {"type": "text", "text": USER_PROMPT}
                        ]}
                    ],
                    "response_format": strict_schema
                }
            }
            out.write(json.dumps(request) + "\n")
            valid_requests += 1

    if valid_requests == 0:
        raise ValueError("CRITICAL ERROR: No valid images found to process. Halting.")
    
    return JSONL_IN

# ─────────────────────────────────────────────
# 2. UPLOAD & LAUNCH
# ─────────────────────────────────────────────
def launch_batch(file_path):
    print("[*] Uploading JSONL to OpenAI...")
    upload = client.files.create(file=open(file_path, "rb"), purpose="batch")
    
    print("[*] Creating Batch Job...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"[*] Batch launched successfully! ID: {batch.id}")
    return batch.id

# ─────────────────────────────────────────────
# 3. AUTO-POLLING LOOP
# ─────────────────────────────────────────────
def wait_for_batch(batch_id):
    print(f"\n[*] Entering auto-polling loop. I will check status every 60 seconds.")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed
        total = batch.request_counts.total
        failed = batch.request_counts.failed
        
        print(f"    [Status: {status}] -> {completed}/{total} processed (Failed: {failed})")
        
        if status == "completed":
            print("\n[*] Batch finished executing successfully!")
            return batch.output_file_id
        elif status in ["failed", "cancelled", "expired"]:
            print(f"\n[!] BATCH HALTED with status: {status}")
            print(f"    Errors: {batch.errors}")
            sys.exit(1)
            
        time.sleep(60)

# ─────────────────────────────────────────────
# 4. DOWNLOAD & PARSE CSV
# ─────────────────────────────────────────────
def download_and_parse(file_id):
    print("[*] Downloading raw results...")
    content = client.files.content(file_id).content
    with open(JSONL_OUT, "wb") as f:
        f.write(content)

    print("[*] Rebuilding metadata and writing final CSV...")
    meta_dict = {r["dcdd_id"]: r for r in csv.DictReader(open(METADATA_CSV, encoding="utf-8"))}
    
    fieldnames = list(list(meta_dict.values())[0].keys()) + list(ImageAttributes.model_fields.keys())

    parsed_count = 0
    with open(JSONL_OUT, "r", encoding="utf-8") as fin, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        for line in fin:
            obj = json.loads(line)
            dcdd_id = obj["custom_id"]
            
            if obj.get("error"):
                print(f"    [WARN] Row {dcdd_id} failed API level: {obj['error']['message']}")
                continue
            if obj["response"]["status_code"] != 200:
                print(f"    [WARN] Row {dcdd_id} HTTP error: {obj['response']['status_code']}")
                continue
            
            attrs = json.loads(obj["response"]["body"]["choices"][0]["message"]["content"])
            writer.writerow({**meta_dict[dcdd_id], **attrs})
            parsed_count += 1
            
    print(f"\n{'='*60}")
    print(f"  MISSION ACCOMPLISHED.")
    print(f"  Successfully extracted {parsed_count} images.")
    print(f"  Final data saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            b_id = sys.argv[1]
            print(f"[*] RESUME MODE ACTIVATED. Reconnecting to Batch: {b_id}")
            output_file_id = wait_for_batch(b_id)
            if output_file_id:
                download_and_parse(output_file_id)
            else:
                print("[!] Batch completed but no output file was generated.")
        else:
            jsonl_path = build_batch()
            b_id = launch_batch(jsonl_path)
            output_file_id = wait_for_batch(b_id)
            if output_file_id:
                download_and_parse(output_file_id)
            else:
                print("[!] Batch completed but no output file was generated.")
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {str(e)}")