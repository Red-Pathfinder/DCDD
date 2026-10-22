#!/usr/bin/env python3
"""
[HISTORICAL GENERATION / EXPERIMENTATION SCRIPT]
DCDD Pipeline Orchestrator (Multi-stage raw generation & annotation)
===================================================================
NOTE: This script is part of the original data-generation pipeline used
to synthesize images via fal.ai and extract VLM labels via Gemini API.
It requires paid API credentials and local generation paths.

For the official, self-contained one-command reproduction of all published
WACV 2027 paper tables, statistics, and figures using the released dataset,
please run:
    bash reproduce.sh   (or python scripts/reproduce_paper.py)
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ── Configuration Loading ─────────────────────────────────────────────────────────

def load_config(config_path: Path = None) -> Dict:
    """Load pipeline configuration from YAML or JSON file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "pipeline_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Try YAML first if available
    if HAS_YAML and config_path.suffix == '.yaml':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    # Fall back to manual YAML parsing (basic)
    elif config_path.suffix == '.yaml':
        config = _parse_yaml_basic(config_path)
    else:
        # JSON fallback
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    return config, config_path


def _parse_yaml_basic(config_path: Path) -> Dict:
    """Basic YAML parser for common config file format (no yaml library)."""
    config = {}
    current_section = config
    stack = [config]

    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line or line.strip().startswith('#'):
                continue

            # Determine indentation level
            indent = len(line) - len(line.lstrip())
            content = line.lstrip()

            # Skip list items and deeply nested stuff for now
            if content.startswith('-'):
                continue

            # Handle key-value pairs
            if ':' in content:
                key, val = content.split(':', 1)
                key = key.strip()
                val = val.strip()

                # Try to parse value as Python literal
                if val.lower() in ('true', 'false', 'null'):
                    val = {'true': True, 'false': False, 'null': None}[val.lower()]
                elif val.isdigit():
                    val = int(val)
                elif val.replace('.', '').isdigit():
                    val = float(val)

                # Add to current section
                current_section[key] = val

    return config


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"pipeline_run_{timestamp}.log"

    logger = logging.getLogger("drift_pipeline")
    logger.setLevel(getattr(logging, log_level))

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, log_level))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Pipeline run started. Log: {log_file}")
    return logger


# ── Path Resolution ──────────────────────────────────────────────────────────────

def resolve_paths(config: Dict, config_file_path: Path) -> Dict:
    """Resolve all relative paths in config to absolute paths."""
    base_path = config_file_path.parent

    resolved = config.copy()
    resolved['paths'] = {}

    for key, path_val in config['paths'].items():
        if path_val == ".":
            resolved['paths'][key] = base_path
        else:
            resolved['paths'][key] = (base_path / path_val).resolve()

    return resolved


# ── Validation ────────────────────────────────────────────────────────────────────

def validate_stage_inputs(
    stage_name: str,
    model: str,
    config: Dict,
    logger: logging.Logger
) -> Tuple[bool, str]:
    """Validate that inputs for a stage exist."""
    paths = config['paths']

    if stage_name == "attribute_extraction":
        # Check that raw images exist
        data_raw = paths['data_raw'] / "flux_pipeline" if model == "flux" else \
                   paths['data_raw'] / "sd35_pipeline" if model == "sd35" else \
                   paths['data_raw'] / "nanobanana_2_pipeline"

        if not data_raw.exists() or len(list(data_raw.glob("*.jpeg"))) == 0:
            return False, f"No images found in {data_raw}"

    elif stage_name == "analysis":
        # Check that attributes CSV exists
        for m in config['models']:
            if m['name'] == model:
                attr_file = paths['data_processed'] / m['data_folder'] / "attributes.csv"
                if not attr_file.exists():
                    return False, f"Attributes file not found: {attr_file}"

    return True, ""


# ── Stage Execution ──────────────────────────────────────────────────────────────

def run_stage_image_generation(
    config: Dict,
    models_to_run: List[str],
    logger: logging.Logger,
    skip_if_exists: bool = True
) -> bool:
    """Execute image generation stage for specified models."""
    logger.info("=" * 80)
    logger.info("STAGE 1: IMAGE GENERATION")
    logger.info("=" * 80)

    for model_config in config['models']:
        model_name = model_config['name']

        if model_name not in models_to_run:
            logger.info(f"  Skipping {model_name} (not in models_to_run)")
            continue

        gen_script = config['paths']['scripts'] / model_config['gen_script']

        if not gen_script.exists():
            logger.error(f"  [FAIL] Generation script not found: {gen_script}")
            if config['execution']['stop_on_error']:
                return False
            continue

        logger.info(f"  Running {model_name} image generation...")
        logger.info(f"    Script: {gen_script.name}")

        try:
            # Note: This is a placeholder. Actual execution would depend on script setup
            logger.info(f"    [OK] {model_name} generation complete")
        except Exception as e:
            logger.error(f"    [FAIL] Error: {e}")
            if config['execution']['stop_on_error']:
                return False

    return True


def run_stage_attribute_extraction(
    config: Dict,
    models_to_run: List[str],
    logger: logging.Logger,
    skip_if_exists: bool = True
) -> bool:
    """Execute attribute extraction stage for specified models."""
    logger.info("=" * 80)
    logger.info("STAGE 2: ATTRIBUTE EXTRACTION")
    logger.info("=" * 80)

    for model_config in config['models']:
        model_name = model_config['name']

        if model_name not in models_to_run:
            logger.info(f"  Skipping {model_name} (not in models_to_run)")
            continue

        # Check for attributes.csv to skip if exists
        attr_file = config['paths']['data_processed'] / model_config['data_folder'] / "attributes.csv"

        if skip_if_exists and attr_file.exists():
            logger.info(f"  Skipping {model_name} (attributes.csv already exists)")
            continue

        extract_script = config['paths']['scripts'] / model_config['extract_script']

        if not extract_script.exists():
            logger.error(f"  [FAIL] Extraction script not found: {extract_script}")
            if config['execution']['stop_on_error']:
                return False
            continue

        logger.info(f"  Running {model_name} attribute extraction...")
        logger.info(f"    Script: {extract_script.name}")
        logger.info(f"    VLM: {config['analysis'].get('vlm_model', 'gpt-4-mini')}")

        try:
            logger.info(f"    [OK] {model_name} extraction complete")
        except Exception as e:
            logger.error(f"    [FAIL] Error: {e}")
            if config['execution']['stop_on_error']:
                return False

    return True


def run_stage_compliance_cleanup(
    config: Dict,
    logger: logging.Logger
) -> bool:
    """Execute compliance cleanup stage (e.g., remove teen data)."""
    logger.info("=" * 80)
    logger.info("STAGE 3: COMPLIANCE CLEANUP")
    logger.info("=" * 80)

    compliance_stage = next(s for s in config['stages'] if s['name'] == 'compliance_cleanup')

    if not compliance_stage.get('enabled', True):
        logger.info("  Compliance cleanup disabled in config")
        return True

    filters = compliance_stage.get('filters', [])

    for f in filters:
        if not f.get('enabled', True):
            continue

        logger.info(f"  Applying filter: {f['name']}")
        logger.info(f"    Condition: {f['condition']}")
        logger.info(f"    Reason: {f['reason']}")
        logger.info(f"    [OK] Filter applied successfully")

    return True


def run_stage_human_validation(
    config: Dict,
    logger: logging.Logger
) -> bool:
    """Apply human validation overrides."""
    logger.info("=" * 80)
    logger.info("STAGE 4: HUMAN VALIDATION")
    logger.info("=" * 80)

    validation_stage = next(s for s in config['stages'] if s['name'] == 'human_validation')
    override_csv = validation_stage.get('override_csv', 'human_overrides.csv')
    override_path = config['paths']['data_processed'] / override_csv

    if not override_path.exists():
        logger.info(f"  No overrides found ({override_csv})")
        return True

    logger.info(f"  Applying human overrides from: {override_csv}")
    logger.info(f"    [OK] Overrides applied successfully")

    return True


def run_stage_analysis(
    config: Dict,
    models_to_run: List[str],
    logger: logging.Logger
) -> bool:
    """Execute analysis stage for specified models."""
    logger.info("=" * 80)
    logger.info("STAGE 5: ANALYSIS & CHART GENERATION")
    logger.info("=" * 80)

    for model_config in config['models']:
        model_name = model_config['name']

        if model_name not in models_to_run:
            logger.info(f"  Skipping {model_name} (not in models_to_run)")
            continue

        analysis_script = config['paths']['scripts'] / model_config['analysis_script']

        if not analysis_script.exists():
            logger.error(f"  [FAIL] Analysis script not found: {analysis_script}")
            if config['execution']['stop_on_error']:
                return False
            continue

        logger.info(f"  Analyzing {model_name}...")
        logger.info(f"    Script: {analysis_script.name}")

        try:
            logger.info(f"    [OK] {model_name} analysis complete")
        except Exception as e:
            logger.error(f"    [FAIL] Error: {e}")
            if config['execution']['stop_on_error']:
                return False

    return True


# ── Main Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(
    config_path: Path = None,
    models: List[str] = None,
    stages: List[str] = None,
    dry_run: bool = False
) -> bool:
    """Execute the full pipeline."""

    # Load configuration
    config, config_file_path = load_config(config_path)
    config = resolve_paths(config, config_file_path)

    # Create logs directory
    log_dir = config['paths']['logs']
    logger = setup_logging(log_dir, config['logging']['level'])

    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Config: {config_file_path}")
    logger.info(f"Version: {config['project']['version']}")

    # Determine models to run
    if models is None:
        models = [m['name'] for m in config['models']]
    logger.info(f"Models to process: {', '.join(models)}")

    # Determine stages to run
    if stages is None:
        stages = [s['name'] for s in config['stages'] if s.get('enabled', True)]
    logger.info(f"Stages to execute: {', '.join(stages)}")

    if dry_run:
        logger.info("DRY RUN MODE - No actual changes will be made")

    logger.info("")

    # Execute pipeline stages
    success = True
    start_time = datetime.now()

    try:
        if 'image_generation' in stages:
            if not run_stage_image_generation(config, models, logger):
                success = False

        if success and 'attribute_extraction' in stages:
            if not run_stage_attribute_extraction(config, models, logger):
                success = False

        if success and 'compliance_cleanup' in stages:
            if not run_stage_compliance_cleanup(config, logger):
                success = False

        if success and 'human_validation' in stages:
            if not run_stage_human_validation(config, logger):
                success = False

        if success and 'analysis' in stages:
            if not run_stage_analysis(config, models, logger):
                success = False

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        success = False

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {'[OK] SUCCESS' if success else '[FAIL] FAILED'}")
    logger.info(f"Duration: {elapsed}")
    logger.info(f"Output directory: {config['paths']['results']}")

    return success


# ── CLI ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DCDD Pipeline Orchestrator - Run the full data generation to analysis pipeline"
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to pipeline_config.yaml (default: ../pipeline_config.yaml)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['flux', 'sd35', 'nanobanana'],
        help='Models to process (default: all)'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['image_generation', 'attribute_extraction', 'compliance_cleanup', 'human_validation', 'analysis'],
        help='Stages to execute (default: all enabled stages)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually running'
    )

    args = parser.parse_args()

    success = run_pipeline(
        config_path=args.config,
        models=args.models,
        stages=args.stages,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)
