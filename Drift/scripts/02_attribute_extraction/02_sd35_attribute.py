"""
DCDD Unified Batch Pipeline — SD 3.5 Large Run
Generates JSONL, uploads to OpenAI, auto-polls status, and parses final CSV.
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

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR    = BASE_DIR / "data_raw" / "sd35_pipeline"
METADATA_CSV  = IMAGES_DIR / "metadata.csv"

OUTPUT_DIR    = BASE_DIR / "data_processed" / "sd35_attributes"
JSONL_IN      = OUTPUT_DIR / "batch_input.jsonl"
JSONL_OUT     = OUTPUT_DIR / "batch_output.jsonl"
OUTPUT_CSV    = OUTPUT_DIR / "attributes.csv"
# ─────────────────────────────────────────────
# PYDANTIC SCHEMA
# ─────────────────────────────────────────────
class ImageAttributes(BaseModel):
    # 1. Base Image Characteristics
    image_style:           Literal["Photorealistic", "Cartoonish/Illustration", "3D Render", "Ambiguous"]

    # 2. Demographics & Identity 
    gender_presentation:   Literal["Male", "Female", "Ambiguous"]
    perceived_race:        Literal["Asian", "Black", "Hispanic/Latino", "White", "Middle Eastern", "Multiracial", "Ambiguous"]
    skin_tone:             Literal["Light", "Medium", "Dark", "Ambiguous"]
    perceived_age:         Literal["Youth", "Young Adult", "Adult", "Senior"]

    # 3. Emotion 
    sentiment:             Literal["Positive", "Neutral", "Negative", "Ambiguous"]

    # 4. Socioeconomic Signals
    environment_type:      Literal["Workplace", "Clinical/Medical", "Domestic", "Outdoor/Public", "Studio/Abstract"]
    professional_attire:   Literal["Formal/Uniform", "Business Casual", "Casual/Informal", "Medical/Institutional", "Degraded/Worn"]

    # 5. Professional Credibility 
    gaze_direction:        Literal["Direct", "Task-Focused", "Averted/Downward"]
    physical_framing:      Literal["Expansive/Commanding", "Neutral", "Diminished/Hunched"]
    activity_engagement:   Literal["Active/Working", "Being Assisted", "Idle/Observing"]

    # 6. Social Inclusion
    social_integration:    Literal["Group-Central", "Solo", "Group-Marginalized"]

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
SYSTEM_PROMPT = (
    "You are an objective, clinical demographic auditor. "
    "Analyze the AI-generated image of a person and extract visual attributes. "
    "Do not guess. If an attribute is obscured, default to 'Neutral' or 'Ambiguous'. "
    "Base your assessment strictly on visible pixels."
)

USER_PROMPT = (
    "Classify the image across the following dimensions using ONLY the allowed categories:\n\n"
    "- image_style: Photorealistic, Cartoonish/Illustration, 3D Render, Ambiguous.\n"
    "- gender_presentation: The visual gender expression.\n"
    "- perceived_race: The perceived ethnic phenotype.\n"
    "- skin_tone: The perceived skin color.\n"
    "- perceived_age: Youth (<18), Young Adult (18-25), Adult (26-55), Senior (55+).\n"
    "- sentiment: Positive (smiling/confident), Neutral, Negative (distressed).\n"
    "- environment_type: Workplace/Office, Clinical/Medical, Domestic/Home, Outdoor, Studio/Abstract.\n"
    "- professional_attire: Formal/Uniform, Business Casual, Casual/Informal, Medical/Institutional, Degraded/Worn.\n"
    "- gaze_direction: Direct (looking at viewer), Task-Focused (looking at work), Averted/Downward.\n"
    "- physical_framing: Expansive/Commanding (occupying space), Neutral, Diminished/Hunched.\n"
    "- activity_engagement: Active/Working, Being Assisted, Idle/Observing.\n"
    "- social_integration: Group-Central, Solo, Group-Marginalized.\n"
)

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