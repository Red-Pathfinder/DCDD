"""
DCDD Stage 1 — Batch API: Script 2 of 2
Retrieve a completed Batch job and parse results into attributes.csv.

Run this after batch_01_prepare_and_upload.py has printed a batch_id.

Usage:
    python batch_02_retrieve_and_parse.py --batch_id batch_abc123xyz
    python batch_02_retrieve_and_parse.py   # uses BATCH_ID variable below
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Fallback batch_id if not passed via CLI. Paste yours here if not using argparse.
BATCH_ID_FALLBACK = ""   # e.g. "batch_abc123xyz"

# Paths (must mirror script 1)
METADATA_CSV  = Path("dcdd_output_flux/metadata.csv")
OUTPUT_DIR    = Path("dcdd_batch_results")
RAW_JSONL_OUT = OUTPUT_DIR / "batch_output_raw.jsonl"
OUTPUT_CSV    = OUTPUT_DIR / "attributes.csv"

# ─────────────────────────────────────────────
# PYDANTIC SCHEMA  (must stay in sync with batch_01)
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

OUTPUT_FIELDNAMES = [
    # ── Original metadata columns ──
    "dcdd_id", "pair_id", "seed", "model", "sector", "profession",
    "age_bin", "image_type", "disability", "filename",
    # ── Extracted attributes ──
    "gender_presentation", "skin_tone", "perceived_age", "environment_type",
    "professional_attire", "gaze_direction", "physical_framing",
    "activity_engagement", "social_integration"
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_batch_id() -> str:
    """
    Resolve batch_id from:
      1. --batch_id CLI argument  (preferred)
      2. BATCH_ID_FALLBACK variable at top of file
    """
    parser = argparse.ArgumentParser(description="Retrieve and parse DCDD Batch API results.")
    parser.add_argument("--batch_id", type=str, default=BATCH_ID_FALLBACK,
                        help="The OpenAI batch_id returned by batch_01_prepare_and_upload.py")
    args = parser.parse_args()

    if not args.batch_id:
        print("ERROR: No batch_id provided. Use --batch_id or set BATCH_ID_FALLBACK in the script.")
        sys.exit(1)

    return args.batch_id


def load_metadata_index(csv_path: Path) -> dict[str, dict]:
    """
    Build a lookup dict: { dcdd_id -> full row dict }
    This lets us join attribute results back to original metadata in O(1).
    """
    index = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[row["dcdd_id"]] = row
    return index


def parse_batch_result_line(line: str) -> tuple[Optional[str], Optional[ImageAttributes], Optional[str]]:
    """
    Parse one line from the Batch API output JSONL.

    Returns: (dcdd_id, ImageAttributes | None, error_message | None)

    Batch output line structure:
    {
      "id": "...",
      "custom_id": "<dcdd_id>",       ← this is what we set in batch_01
      "response": {
        "status_code": 200,
        "body": {                       ← standard chat.completions response body
          "choices": [{
            "message": {
              "content": "<json string>"  ← the structured output we want
            }
          }]
        }
      },
      "error": null | { "code": ..., "message": ... }
    }
    """
    obj = json.loads(line)
    dcdd_id = obj.get("custom_id")

    # ── Case 1: top-level API error (e.g. rate limit, content filter) ─────────
    if obj.get("error"):
        err_msg = obj["error"].get("message", "Unknown batch-level error")
        return dcdd_id, None, f"Batch error: {err_msg}"

    # ── Case 2: HTTP-level error inside response ───────────────────────────────
    response = obj.get("response", {})
    status   = response.get("status_code")
    if status != 200:
        return dcdd_id, None, f"HTTP {status}: {response}"

    # ── Case 3: Parse the content string as JSON → validate with Pydantic ─────
    try:
        content_str = response["body"]["choices"][0]["message"]["content"]
        content_dict = json.loads(content_str)
        attrs = ImageAttributes(**content_dict)
        return dcdd_id, attrs, None

    except (KeyError, json.JSONDecodeError, ValidationError) as e:
        return dcdd_id, None, f"Parse/validation error: {e}"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch_id = get_batch_id()

    print("\n" + "=" * 60)
    print(f"DCDD BATCH RETRIEVAL")
    print(f"batch_id : {batch_id}")
    print("=" * 60)

    # ── 1. Check batch status ─────────────────────────────────────────────────
    batch = client.batches.retrieve(batch_id)
    status = batch.status

    print(f"\nStatus   : {status}")
    print(f"Progress : {batch.request_counts.completed} completed / "
          f"{batch.request_counts.total} total / "
          f"{batch.request_counts.failed} failed")

    # Possible statuses: validating, in_progress, finalizing, completed, failed, expired, cancelled
    if status != "completed":
        print(f"\nBatch is not yet completed (current: '{status}').")
        print("Re-run this script once the batch completes. Typical time: minutes to hours.")
        if status == "failed":
            print(f"Failure details: {batch.errors}")
        return

    if not batch.output_file_id:
        print("\nERROR: Batch completed but output_file_id is missing. Check OpenAI dashboard.")
        return

    # ── 2. Download raw output JSONL ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading output file: {batch.output_file_id} ...")
    file_content = client.files.content(batch.output_file_id)

    with open(RAW_JSONL_OUT, "wb") as f:
        f.write(file_content.content)

    print(f"Saved raw output to    : {RAW_JSONL_OUT}")

    # ── 3. Load metadata index for join ──────────────────────────────────────
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_CSV}")

    metadata_index = load_metadata_index(METADATA_CSV)
    print(f"Metadata index loaded  : {len(metadata_index)} rows")

    # ── 4. Parse JSONL and write attributes.csv ───────────────────────────────
    parsed_count = 0
    failed_count = 0
    missing_meta = 0

    with open(RAW_JSONL_OUT, "r", encoding="utf-8") as jsonl_file, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue

            dcdd_id, attrs, error = parse_batch_result_line(line)

            # ── Failed request ─────────────────────────────────────────────
            if error:
                print(f"  [FAIL] {dcdd_id}: {error}")
                failed_count += 1
                continue

            # ── Metadata join ──────────────────────────────────────────────
            meta = metadata_index.get(dcdd_id)
            if not meta:
                print(f"  [WARN] No metadata found for dcdd_id: {dcdd_id} — skipping.")
                missing_meta += 1
                continue

            # ── Write the merged row ───────────────────────────────────────
            out_row = {
                # Original metadata
                "dcdd_id":    dcdd_id,
                "pair_id":    meta.get("pair_id", ""),
                "seed":       meta.get("seed", ""),
                "model":      meta.get("model", ""),
                "sector":     meta.get("sector", ""),
                "profession": meta.get("profession", ""),
                "age_bin":    meta.get("age_bin", ""),
                "image_type": meta.get("image_type", ""),
                "disability": meta.get("disability", ""),
                "filename":   meta.get("filename", ""),
                # Extracted attributes
                "gender_presentation":  attrs.gender_presentation,
                "skin_tone":            attrs.skin_tone,
                "perceived_age":        attrs.perceived_age,
                "environment_type":     attrs.environment_type,
                "professional_attire":  attrs.professional_attire,
                "gaze_direction":       attrs.gaze_direction,
                "physical_framing":     attrs.physical_framing,
                "activity_engagement":  attrs.activity_engagement,
                "social_integration":   attrs.social_integration,
            }
            writer.writerow(out_row)
            parsed_count += 1

            print(f"  ✓ {dcdd_id} | {attrs.gender_presentation} | {attrs.skin_tone} | {attrs.gaze_direction}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Parsing complete!")
    print(f"  Rows written  : {parsed_count}")
    print(f"  Failed        : {failed_count}")
    print(f"  Missing meta  : {missing_meta}")
    print(f"  Output CSV    : {OUTPUT_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
