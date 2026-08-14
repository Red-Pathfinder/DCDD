#!/usr/bin/env python3
"""
DCDD Pipeline Stage Output Validator
Validates that outputs from each pipeline stage are consistent and complete
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import csv

# ── Setup ─────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger("drift_validator")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)-8s | %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ── Validators ────────────────────────────────────────────────────────────────────

def validate_attributes_csv(csv_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """Validate attributes.csv file structure and content."""

    if not csv_path.exists():
        return False, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Cannot parse CSV: {e}"

    # Expected columns
    required_columns = [
        "image_id", "image_type",
        "gender_presentation", "perceived_race", "skin_tone", "perceived_age",
        "sentiment", "environment_type", "professional_attire", "gaze_direction",
        "physical_framing", "activity_engagement", "social_integration", "image_style"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"

    # Check for empty rows
    if len(df) == 0:
        return False, "CSV is empty"

    # Check image_type values
    valid_types = {"neutral", "conditioned"}
    invalid_types = set(df['image_type'].str.lower().unique()) - valid_types
    if invalid_types:
        return False, f"Invalid image_type values: {invalid_types}"

    # Check for missing values in critical columns
    critical_cols = ["image_id", "image_type"]
    missing_values = df[critical_cols].isnull().sum()
    if missing_values.sum() > 0:
        return False, f"Missing critical values: {missing_values[missing_values > 0].to_dict()}"

    return True, f"✓ Valid ({len(df)} rows)"


def validate_human_overrides_csv(csv_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """Validate human_overrides.csv file structure."""

    if not csv_path.exists():
        return True, "No human_overrides.csv (optional)"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Cannot parse CSV: {e}"

    required_columns = ["image_id", "attribute", "original_prediction", "override_value"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"

    if len(df) == 0:
        return True, "Overrides file is empty (no manual corrections)"

    return True, f"✓ Valid ({len(df)} overrides)"


def validate_image_count_consistency(
    image_dir: Path,
    attributes_csv: Path,
    logger: logging.Logger
) -> Tuple[bool, str]:
    """Check that image count matches attributes CSV count."""

    if not image_dir.exists():
        return False, f"Image directory not found: {image_dir}"

    if not attributes_csv.exists():
        return False, f"Attributes CSV not found: {attributes_csv}"

    try:
        image_files = list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_count = len(image_files)

        df = pd.read_csv(attributes_csv)
        csv_count = len(df)

        if image_count != csv_count:
            return False, f"Mismatch: {image_count} images vs {csv_count} attribute rows"

        return True, f"✓ Counts match ({image_count} images)"

    except Exception as e:
        return False, f"Validation error: {e}"


def validate_neural_age_compliance(csv_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    """Check that teen data (age 13-17) has been removed per compliance."""

    if not csv_path.exists():
        return False, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Cannot parse CSV: {e}"

    # Check for teen data - if age_bin column exists
    if 'age_bin' in df.columns:
        teen_rows = df[df['age_bin'].str.contains('13-17|teen', case=False, na=False)]
        if len(teen_rows) > 0:
            return False, f"Teen data found ({len(teen_rows)} rows) - compliance violation"

    # Also check perceived_age column
    if 'perceived_age' in df.columns:
        teen_values = {'teen', '13-17', '13-17 teen'}
        teen_rows = df[df['perceived_age'].isin(teen_values)]
        if len(teen_rows) > 0:
            return False, f"Teen data found in perceived_age ({len(teen_rows)} rows)"

    return True, "✓ No teen data (compliance OK)"


# ── Comprehensive Validators ──────────────────────────────────────────────────────

def validate_model_attributes(
    base_path: Path,
    model_name: str,
    data_folder: str,
    image_folder: str,
    logger: logging.Logger
) -> bool:
    """Validate all outputs for a single model."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Paths
    attributes_csv = base_path / data_folder / "attributes.csv"
    image_dir = base_path / "data_raw" / image_folder
    overrides_csv = base_path / "data_processed" / "human_overrides.csv"

    all_valid = True

    # 1. Attributes CSV structure
    logger.info(f"  Checking {data_folder}/attributes.csv...")
    valid, msg = validate_attributes_csv(attributes_csv, logger)
    logger.info(f"    {msg}")
    all_valid = all_valid and valid

    # 2. Image count consistency
    if image_dir.exists():
        logger.info(f"  Checking image count consistency...")
        valid, msg = validate_image_count_consistency(image_dir, attributes_csv, logger)
        logger.info(f"    {msg}")
        all_valid = all_valid and valid

    # 3. Compliance: no teen data
    logger.info(f"  Checking compliance (no teen data)...")
    valid, msg = validate_neural_age_compliance(attributes_csv, logger)
    logger.info(f"    {msg}")
    all_valid = all_valid and valid

    return all_valid


def validate_global_state(base_path: Path, logger: logging.Logger) -> bool:
    """Validate global state (overrides, etc.)."""

    logger.info(f"\n{'='*60}")
    logger.info("Validating Global State")
    logger.info(f"{'='*60}")

    all_valid = True

    # Check human_overrides.csv
    overrides_csv = base_path / "data_processed" / "human_overrides.csv"
    logger.info(f"  Checking human_overrides.csv...")
    valid, msg = validate_human_overrides_csv(overrides_csv, logger)
    logger.info(f"    {msg}")
    all_valid = all_valid and valid

    return all_valid


# ── Main ──────────────────────────────────────────────────────────────────────────

def validate_all(base_path: Path = None) -> bool:
    """Run all validations."""

    logger = setup_logging()

    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent

    logger.info(f"DCDD Pipeline Validator")
    logger.info(f"Base path: {base_path}")

    # Models to validate
    models = [
        {
            "name": "flux",
            "data_folder": "data_processed/01_flux_attributes",
            "image_folder": "flux_pipeline"
        },
        {
            "name": "sd35",
            "data_folder": "data_processed/02_sd35_attributes",
            "image_folder": "sd35_pipeline"
        },
        {
            "name": "nanobanana",
            "data_folder": "data_processed/03_nanobanana_attributes",
            "image_folder": "nanobanana_2_pipeline"
        }
    ]

    all_valid = True

    # Validate each model
    for model in models:
        valid = validate_model_attributes(
            base_path / "Drift",
            model['name'],
            model['data_folder'],
            model['image_folder'],
            logger
        )
        all_valid = all_valid and valid

    # Validate global state
    valid = validate_global_state(base_path / "Drift", logger)
    all_valid = all_valid and valid

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation Summary: {'✓ ALL PASS' if all_valid else '✗ FAILURES FOUND'}")
    logger.info(f"{'='*60}")

    return all_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate DCDD pipeline stage outputs"
    )
    parser.add_argument(
        '--base-path',
        type=Path,
        help='Base path to Research directory (default: parent of scripts dir)'
    )

    args = parser.parse_args()
    success = validate_all(args.base_path)
    sys.exit(0 if success else 1)
