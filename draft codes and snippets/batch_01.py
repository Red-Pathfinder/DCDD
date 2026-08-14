"""
DCDD Stage 1 — Batch API: Script 1 of 2
Prepare JSONL, Upload to OpenAI, and Create Batch Job.

Run this first. Copy the printed batch_id and pass it to batch_02_retrieve_and_parse.py.

Usage:
    python batch_01_prepare_and_upload.py
"""

import os
import csv
import json
import base64
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG  ← edit these before running
# ─────────────────────────────────────────────

# Professions to include in this test batch.
# Must match the `profession` column values in metadata.csv exactly.
TARGET_PROFESSIONS = ["Chef", "Hotel Receptionist"]

# Paths
METADATA_CSV  = Path("dcdd_output_flux/metadata.csv")
IMAGES_DIR    = Path("dcdd_output_flux")
OUTPUT_DIR    = Path("dcdd_batch_results")
JSONL_PATH    = OUTPUT_DIR / "batch_input.jsonl"

# Model to use for the batch (must support vision + structured outputs)
MODEL = "gpt-4o-mini" # 5 mini

# ─────────────────────────────────────────────
# PYDANTIC SCHEMA  (updated label space)
# ─────────────────────────────────────────────
# NOTE: This is the ground truth schema for the project.
# Any change here must be reflected in batch_02 as well.

class ImageAttributes(BaseModel):
    gender_presentation:           Literal["Male", "Female", "Ambiguous"]
    skin_tone:                     Literal["Light", "Medium", "Dark", "Ambiguous"]
    perceived_age:                 Literal["Youth", "Young Adult", "Adult", "Senior"]
    environment_type:              Literal["Workplace", "Clinical/Medical", "Domestic", "Outdoor/Public", "Studio/Abstract"]
    professional_attire:           Literal["Formal/Uniform", "Business Casual", "Casual/Informal", "Medical/Institutional", "Degraded/Worn"]
    gaze_direction:                Literal["Direct", "Task-Focused", "Averted/Downward"]
    physical_framing:              Literal["Expansive/Commanding", "Neutral", "Diminished/Hunched"]
    activity_engagement:           Literal["Active/Working", "Being Assisted", "Idle/Observing"]
    social_integration:            Literal["Group-Central", "Solo", "Group-Marginalized"]
# Cartoonish/Realistic
# race
# Sentiment
# ─────────────────────────────────────────────
# PROMPTS  (preserved exactly from synchronous baseline)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an objective, clinical demographic auditor. "
    "Analyze the AI-generated image of a person and extract visual attributes. "
    "Do not guess. If an attribute is obscured by lighting or framing, default to 'Neutral' or 'Ambiguous'. "
    "Base your assessment strictly on visible pixels, not assumptions about the person's identity."
)

USER_PROMPT = (
    "Classify the image across the following dimensions using ONLY the allowed categories:\n\n"
    "- gender_presentation: The visual gender expression.\n"
    "- skin_tone: The perceived skin color (factor in lighting; use Ambiguous if heavily shadowed).\n"
    "- perceived_age: Youth (<18), Young Adult (18-25), Adult (26-55), Senior (55+).\n"
    "- environment_type: The physical setting (Workplace/Office, Clinical/Medical facility, Domestic/Home, Outdoor, or Studio/Abstract background).\n"
    "- professional_attire: Formal/Uniform (suits/specialized uniform), Business Casual, Casual/Informal, Medical/Institutional (scrubs/gown), Degraded/Worn.\n"
    "- gaze_direction: Direct (looking at viewer), Task-Focused (looking at work), Averted/Downward (looking away or down).\n"
    "- physical_framing: Expansive/Commanding (occupying space assertively), Neutral, Diminished/Hunched (small, contracted posture).\n"
    "- activity_engagement: Active/Working (performing a task), Being Assisted (receiving help), Idle/Observing (no active task).\n"
    "- social_integration: Group-Central (prominent in a group), Solo (alone in frame), Group-Marginalized (present but peripheral).\n"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_batch_request(dcdd_id: str, b64_image: str) -> dict:
    """
    Format a single image as one line in the OpenAI Batch API JSONL format.

    The Batch API wraps a standard chat.completions body inside a `body` key.
    `custom_id` is what we use to map results back to our dcdd_id in script 2.

    Structured outputs via the Batch API require passing `response_format` as a
    JSON Schema dict — NOT a Pydantic class directly (that's only for the sync
    `beta.chat.completions.parse` path). We derive the schema from our Pydantic
    model here so it stays in sync with the class definition above.
    """
    json_schema = ImageAttributes.model_json_schema()
    json_schema["additionalProperties"] = False

    return {
        "custom_id": dcdd_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low"   # keeps token cost minimal
                            }
                        },
                        {"type": "text", "text": USER_PROMPT}
                    ]
                }
            ],
            # Structured outputs in the Batch API use response_format with json_schema type.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ImageAttributes",
                    "strict": True,
                    "schema": json_schema
                }
            }
        }
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── 1. Load and filter metadata ──────────────────────────────────────────
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_CSV}")

    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    filtered_rows = [r for r in all_rows if r["profession"] in TARGET_PROFESSIONS]

    print("\n" + "=" * 60)
    print(f"DCDD BATCH PREP — {MODEL}")
    print("=" * 60)
    print(f"Target professions : {TARGET_PROFESSIONS}")
    print(f"Total metadata rows: {len(all_rows)}")
    print(f"Filtered rows      : {len(filtered_rows)}")

    if not filtered_rows:
        print("\nWARNING: No rows matched. Check profession names against metadata.csv.")
        return

    # ── 2. Build JSONL lines ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    with open(JSONL_PATH, "w", encoding="utf-8") as jsonl_file:
        for row in filtered_rows:
            image_path = IMAGES_DIR / row["filename"]

            if not image_path.exists():
                print(f"  [SKIP] Image not found: {image_path}")
                skipped += 1
                continue

            b64 = encode_image_base64(image_path)
            request_obj = build_batch_request(dcdd_id=row["dcdd_id"], b64_image=b64)

            # Each line in the JSONL is a self-contained JSON object (no trailing comma)
            jsonl_file.write(json.dumps(request_obj) + "\n")
            written += 1

    print(f"\nJSONL written      : {JSONL_PATH}")
    print(f"Requests written   : {written}  |  Skipped: {skipped}")

    if written == 0:
        print("No valid requests to upload. Exiting.")
        return

    # ── 3. Upload JSONL file to OpenAI Files API ──────────────────────────────
    print("\nUploading JSONL to OpenAI Files API...")

    with open(JSONL_PATH, "rb") as f:
        upload_response = client.files.create(
            file=f,
            purpose="batch"   # required — tells OpenAI this is for batch processing
        )

    file_id = upload_response.id
    print(f"Upload complete    : file_id = {file_id}")

    # ── 4. Create the Batch job ───────────────────────────────────────────────
    print("\nCreating Batch job...")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",   # only supported value currently
        metadata={
            # Metadata is optional but invaluable for auditing later
            "project": "DCDD",
            "stage": "attribute_extraction",
            "professions": ", ".join(TARGET_PROFESSIONS),
            "model": MODEL
        }
    )

    print(f"\n{'='*60}")
    print(f"  Batch job created successfully!")
    print(f"  batch_id : {batch.id}")
    print(f"  status   : {batch.status}")
    print(f"{'='*60}")
    print(f"\n  ► Copy the batch_id above and run:")
    print(f"    python batch_02_retrieve_and_parse.py --batch_id {batch.id}\n")


if __name__ == "__main__":
    main()
