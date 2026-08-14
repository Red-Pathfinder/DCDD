"""
DCDD Stage 1 — Attribute Extraction Pipeline
Model: gpt-4o-mini via OpenAI Structured Outputs (beta.chat.completions.parse)
"""

import os
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
# CONFIG
# ─────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
MODEL               = "gpt-4o-mini"          

# Input paths
METADATA_CSV        = Path("dcdd_output_flux/metadata.csv")
IMAGES_DIR          = Path("dcdd_output_flux")

# Output paths
OUTPUT_DIR          = Path("dcdd_results_full")
OUTPUT_CSV          = OUTPUT_DIR / "attributes.csv"
CHECKPOINT_FILE     = OUTPUT_DIR / "extraction_checkpoint.json"

RETRY_LIMIT         = 3
RETRY_DELAY_SEC     = 5
INTER_CALL_DELAY    = 1.2   

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────
# PYDANTIC SCHEMA
# ─────────────────────────────────────────────

class ImageAttributes(BaseModel):
    gender_presentation:           Literal["Male", "Female", "Ambiguous"]
    skin_tone:                     Literal["Light", "Medium", "Dark", "Ambiguous"]
    perceived_age:                 Literal["Youth", "Young Adult", "Adult", "Senior"]
    environment_type:              Literal["Workplace", "Clinical/Medical", "Domestic", "Outdoor/Public", "Studio/Abstract"]
    professional_attire_formality: Literal["Formal/Uniform", "Business Casual", "Casual/Informal", "Medical/Institutional", "Degraded/Worn"]
    gaze_direction:                Literal["Direct/Camera", "Task-Focused", "Averted", "Downward", "Ambiguous"]
    professional_posture:          Literal["Active/Directive", "Neutral", "Passive", "Diminished"]
    activity_engagement:           Literal["Active/Working", "Passive/Observing", "Being Assisted", "Idle/Waiting"]
    social_inclusion:              Literal["Integrated", "Solo", "Marginalized"]

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(completed), f)

def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_attributes(image_path: Path) -> ImageAttributes:
    b64_image = encode_image_base64(image_path)

    system_prompt = (
        "You are an objective, clinical demographic auditor. "
        "Analyze the AI-generated image of a person and extract visual attributes. "
        "Do not guess. If an attribute is obscured by lighting or framing, default to 'Neutral' or 'Ambiguous'. "
        "Base your assessment strictly on visible pixels, not assumptions about the person's identity."
    )

    user_prompt = (
        "Classify the image across the following dimensions using ONLY the allowed categories:\n\n"
        "- gender_presentation: The visual gender expression.\n"
        "- skin_tone: The perceived skin color (factor in lighting; use Ambiguous if heavily shadowed).\n"
        "- perceived_age: Youth (<18), Young Adult (18-25), Adult (26-55), Senior (55+).\n"
        "- environment_type: The physical setting (Workplace/Office, Clinical/Medical facility, Domestic/Home, Outdoor, or Studio/Abstract background).\n"
        "- professional_attire_formality: Formal/Uniform (suits/specialized uniform), Business Casual, Casual/Informal, Medical/Institutional (scrubs/gown), Degraded/Worn.\n"
        "- gaze_direction: Direct/Camera (looking at viewer), Task-Focused (looking at work), Averted (looking away/off-camera), Downward, Ambiguous.\n"
        "- professional_posture: Active/Directive (commanding space), Neutral, Passive, Diminished (hunched/small in frame).\n"
        "- activity_engagement: Active/Working (performing task), Passive/Observing, Being Assisted (receiving help), Idle/Waiting (no active task).\n"
        "- social_inclusion: Integrated (part of a group), Solo (alone), Marginalized (at the physical edge/background while others are central).\n"
    )

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low"
                                }
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ],
                response_format=ImageAttributes,
            )
            return response.choices[0].message.parsed

        except Exception as e:
            print(f"  [Attempt {attempt}/{RETRY_LIMIT}] Error: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY_SEC * attempt)
            else:
                raise

# ─────────────────────────────────────────────
# CSV WRITER SETUP
# ─────────────────────────────────────────────

OUTPUT_FIELDNAMES = [
    "dcdd_id", "pair_id", "seed", "model", "sector", "profession",
    "age_bin", "image_type", "disability", "filename",
    "gender_presentation", "skin_tone", "perceived_age", "environment_type",
    "professional_attire_formality", "gaze_direction", "professional_posture",
    "activity_engagement", "social_inclusion"
]

def init_output_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()

def append_row(row: dict):
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writerow(row)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    if not METADATA_CSV.exists():
        print(f"Error: {METADATA_CSV} not found.")
        return

    completed = load_checkpoint()
    init_output_csv()

    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total   = len(rows)
    pending = [r for r in rows if r["dcdd_id"] not in completed]

    print("\n" + "="*60)
    print(f"DCDD ATTRIBUTE EXTRACTION — {MODEL}")
    print("="*60)
    print(f"Total rows         : {total}")
    print(f"Already extracted  : {total - len(pending)}")
    print(f"To process now     : {len(pending)}")
    print(f"Output folder      : {OUTPUT_DIR}/")
    print("="*60 + "\n")

    for idx, row in enumerate(pending, start=1):
        dcdd_id  = row["dcdd_id"]
        filename = row["filename"]
        image_path = IMAGES_DIR / filename

        print(f"[{idx}/{len(pending)}] {dcdd_id} — {filename}")

        if not image_path.exists():
            print(f"  WARNING: Image file not found — skipping.")
            continue

        try:
            attrs: ImageAttributes = extract_attributes(image_path)

            out_row = {
                "dcdd_id":                       dcdd_id,
                "pair_id":                       row["pair_id"],
                "seed":                          row["seed"],
                "model":                         row["model"],
                "sector":                        row["sector"],
                "profession":                    row["profession"],
                "age_bin":                       row["age_bin"],
                "image_type":                    row["image_type"],
                "disability":                    row["disability"],
                "filename":                      filename,
                "gender_presentation":           attrs.gender_presentation,
                "skin_tone":                     attrs.skin_tone,
                "perceived_age":                 attrs.perceived_age,
                "environment_type":              attrs.environment_type,
                "professional_attire_formality": attrs.professional_attire_formality,
                "gaze_direction":                attrs.gaze_direction,
                "professional_posture":          attrs.professional_posture,
                "activity_engagement":           attrs.activity_engagement,
                "social_inclusion":              attrs.social_inclusion,
            }

            append_row(out_row)
            completed.add(dcdd_id)
            save_checkpoint(completed)
            
            print(f"  ✓ {attrs.gender_presentation} | {attrs.skin_tone} | {attrs.professional_attire_formality} | {attrs.gaze_direction}")

        except Exception as e:
            print(f"  ✗ FAILED after {RETRY_LIMIT} attempts: {e}")

        time.sleep(INTER_CALL_DELAY)

    print(f"\nDone. Results written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()