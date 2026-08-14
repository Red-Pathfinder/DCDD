import os
import sys
import csv
import json
import base64
import time
import io
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini" 

NANOBANANA_DIR = Path(r"C:\Study\Coding\Research\Drift\data_raw\nanobanana_2_pipeline")
METADATA_CSV   = NANOBANANA_DIR / "batch_metadata_tracking.csv"
IMAGES_DIR     = NANOBANANA_DIR

OUTPUT_DIR     = Path(r"C:\Study\Coding\Research\Drift\data_processed\02_nanobanana_attributes")
JSONL_IN       = OUTPUT_DIR / "nanobanana_batch_input.jsonl"
JSONL_OUT      = OUTPUT_DIR / "nanobanana_batch_output.jsonl"
OUTPUT_CSV     = OUTPUT_DIR / "nanobanana_attributes_final.csv"

class ImageAttributes(BaseModel):
    image_style:         Literal["Photorealistic", "Stylized/synthetic", "Unclear / mixed style", "Safety Refusal"]
    gender_presentation: Literal["Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation", "Safety Refusal"]
    perceived_race:      Literal["White presenting", "Non-white presenting", "Ambiguous / indeterminate", "Safety Refusal"]
    skin_tone:           Literal["Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate", "Safety Refusal"]
    sentiment:           Literal["Positive", "Negative", "Neutral/unclear", "Safety Refusal"]
    gaze_direction:      Literal["Engaged", "Disengaged", "Unclear / indeterminate", "Safety Refusal"]
    physical_framing:    Literal["Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate", "Safety Refusal"]
    activity_engagement: Literal["Active engagement", "Passive/dependent engagement", "Unclear / indeterminate", "Safety Refusal"]
    social_integration:  Literal["Socially integrated", "Socially isolated", "Unclear", "Safety Refusal"]

strict_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_attributes_extraction",
        "strict": True,
        "schema": ImageAttributes.model_json_schema()
    }
}
strict_schema["json_schema"]["schema"]["additionalProperties"] = False 

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

gender_presentation:
"Masculine presenting"
"Feminine presenting"
"Ambiguous / indeterminate presentation"

perceived_race:
"White presenting"
"Non-white presenting"
"Ambiguous / indeterminate"

skin_tone:
"Lighter skin tone"
"Darker skin tone"
"Ambiguous / indeterminate"

sentiment:
"Positive"
"Negative"
"Neutral/unclear"

gaze_direction:
"Engaged"
"Disengaged"
"Unclear / indeterminate"

physical_framing:
"Expansive/upright posture"
"Constricted/hunched posture"
"Unclear / indeterminate"

activity_engagement:
"Active engagement"
"Passive/dependent engagement"
"Unclear / indeterminate"

social_integration:
"Socially integrated"
"Socially isolated"
"Unclear"
"""

def compress_and_encode_image(img_path):
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 768x768 is the optimal tile size for gpt-4o-mini
        img.thumbnail((768, 768)) 
        
        buffer = io.BytesIO()
        # Drop quality to 75 to crush the file size safely under 200MB
        img.save(buffer, format="JPEG", quality=75) 
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def build_batch():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[*] Prepping {len(rows)} potential images...")
    valid_requests = 0
    missing_images = 0

    with open(JSONL_IN, "w", encoding="utf-8") as out:
        for row in rows:
            custom_id = row["custom_id"]
            img_path = IMAGES_DIR / f"{custom_id}.png"
            
            if not img_path.exists(): 
                missing_images += 1
                continue 
            
            b64 = compress_and_encode_image(img_path)

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": USER_PROMPT}
                        ]}
                    ],
                    "response_format": strict_schema
                }
            }
            out.write(json.dumps(request) + "\n")
            valid_requests += 1

    print(f"    -> {valid_requests} images encoded.")
    print(f"    -> {missing_images} safety blocks skipped.")
    
    # Calculate and print payload size
    file_size_bytes = os.path.getsize(JSONL_IN)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"    -> Payload File Size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 195:
        print("[!] WARNING: File size is dangerously close to the 200MB limit.")

    if valid_requests == 0:
        raise ValueError("CRITICAL ERROR: No valid images found.")
    return JSONL_IN

def launch_batch(file_path):
    print("\n[*] Uploading JSONL to OpenAI...")
    max_retries = 10
    upload = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    -> Upload Attempt {attempt}/{max_retries}...")
            upload = client.files.create(file=open(file_path, "rb"), purpose="batch")
            print("  ✓ Upload successful!")
            break 
        except Exception as e:
            print(f"  ✗ Connection error: {str(e)}")
            if attempt == max_retries:
                raise Exception("CRITICAL: Maximum upload retries reached.")
            time.sleep(15)

    print("[*] Creating Batch Job...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"[*] Batch ID: {batch.id}")
    return batch.id

def wait_for_batch(batch_id):
    print(f"\n[*] Polling status every 60 seconds...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed
        total = batch.request_counts.total
        failed = batch.request_counts.failed
        
        print(f"    [Status: {status}] -> {completed}/{total} processed (Failed: {failed})")
        
        if status == "completed":
            print("\n[*] Batch finished!")
            return batch.output_file_id
        elif status in ["failed", "cancelled", "expired"]:
            print(f"\n[!] BATCH HALTED: {status}\nErrors: {batch.errors}")
            sys.exit(1)
            
        time.sleep(60)

def download_and_parse(file_id):
    print("\n[*] Downloading results...")
    content = client.files.content(file_id).content
    with open(JSONL_OUT, "wb") as f:
        f.write(content)

    print("[*] Writing final CSV...")
    meta_dict = {r["custom_id"]: r for r in csv.DictReader(open(METADATA_CSV, encoding="utf-8"))}
    
    openai_results = {}
    with open(JSONL_OUT, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            cid = obj["custom_id"]
            if obj.get("error") or obj["response"]["status_code"] != 200:
                continue
            openai_results[cid] = json.loads(obj["response"]["body"]["choices"][0]["message"]["content"])

    attribute_keys = list(ImageAttributes.model_fields.keys())
    fieldnames = list(list(meta_dict.values())[0].keys()) + attribute_keys

    parsed = 0
    refused = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        for custom_id, original_meta in meta_dict.items():
            if custom_id in openai_results:
                writer.writerow({**original_meta, **openai_results[custom_id]})
                parsed += 1
            else:
                refused_attrs = {key: "Safety Refusal" for key in attribute_keys}
                writer.writerow({**original_meta, **refused_attrs})
                refused += 1
            
    print(f"\n[*] DONE. Graded: {parsed} | Refusals: {refused} | Total: {parsed + refused}")
    print(f"[*] Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        b_id = sys.argv[1]
        print(f"[*] RESUME MODE: {b_id}")
        output_file_id = wait_for_batch(b_id)
        if output_file_id: download_and_parse(output_file_id)
    else:
        jsonl_path = build_batch()
        b_id = launch_batch(jsonl_path)
        output_file_id = wait_for_batch(b_id)
        if output_file_id: download_and_parse(output_file_id)