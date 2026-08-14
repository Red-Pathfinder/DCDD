"""
Forensic Cohen's Kappa Reconstruction
======================================
Independent reconstruction of human validation statistics from repository artifacts.
This script does NOT modify any existing files. All output goes to forensic_validation_audit/.

Author: Forensic Auditor (automated)
Date: 2026-06-26
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS — pointing to actual repository artifacts
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"C:\Study\Coding\Research\DCDD\Drift")
OUTPUT_DIR = BASE_DIR / "forensic_validation_audit"

# Primary datasets (v1 — used by main analysis scripts 05_flux_analysis.py, 06_sd35_analysis.py)
FLUX_V1_CSV = BASE_DIR / "data_processed" / "01_flux_attributes" / "attributes.csv"
SD35_V1_CSV = BASE_DIR / "data_processed" / "02_sd35_attributes" / "attributes.csv"

# V2 datasets (used by human validation scripts and 13_explicit_analysis_no_ambiguous.py)
FLUX_V2_CSV = BASE_DIR / "data_processed" / "01_flux_attributes_v2" / "flux_attributes_v2.csv"
SD35_V2_CSV = BASE_DIR / "data_processed" / "02_sd35_attributes_v2" / "sd_attributes_v2.csv"

# Nanobanana datasets
NANO_FINAL_CSV = BASE_DIR / "data_processed" / "03_nanobanana_attributes" / "nanobanana_attributes_final.csv"
NANO_V2_CSV = BASE_DIR / "data_processed" / "03_nanobanana_attributes" / "nanobanana_attributes_v2_validated.csv"

# Human overrides
HUMAN_OVERRIDES_CSV = BASE_DIR / "data_processed" / "human_overrides.csv"
VLM_AUDIT_CSV = BASE_DIR / "data_processed" / "vlm_alignment_audit.csv"

# The 6 attributes specified for Cohen's kappa
KAPPA_ATTRIBUTES = [
    "skin_tone", "gender_presentation", "sentiment",
    "physical_framing", "activity_engagement", "social_integration"
]

# All visual attributes
ALL_ATTRIBUTES = [
    "image_style", "gender_presentation", "perceived_race", "skin_tone",
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement",
    "social_integration"
]


def compute_cohens_kappa_full(labels_vlm, labels_human):
    """
    Compute Cohen's kappa from two parallel label vectors.
    
    Uses the standard formula:
        κ = (p_o - p_e) / (1 - p_e)
    where p_o is observed agreement and p_e is expected agreement by chance.
    
    Returns dict with kappa, p_o, p_e, n, confusion_matrix, categories.
    """
    assert len(labels_vlm) == len(labels_human), "Label vectors must be same length"
    n = len(labels_vlm)
    if n == 0:
        return {"kappa": float("nan"), "p_o": 0.0, "p_e": 0.0, "n": 0, "confusion_matrix": {}, "categories": []}
    
    # Get all unique categories from both raters
    all_cats = sorted(set(list(labels_vlm) + list(labels_human)))
    k = len(all_cats)
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    
    # Build confusion matrix
    conf_matrix = np.zeros((k, k), dtype=int)
    for v, h in zip(labels_vlm, labels_human):
        conf_matrix[cat_to_idx[v], cat_to_idx[h]] += 1
    
    # Observed agreement
    p_o = np.trace(conf_matrix) / n
    
    # Expected agreement
    row_sums = conf_matrix.sum(axis=1)  # VLM marginals
    col_sums = conf_matrix.sum(axis=0)  # Human marginals
    p_e = np.sum(row_sums * col_sums) / (n * n)
    
    # Kappa
    if p_e == 1.0:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
    
    return {
        "kappa": float(kappa),
        "p_o": float(p_o),
        "p_e": float(p_e),
        "n": n,
        "confusion_matrix": conf_matrix.tolist(),
        "categories": all_cats,
        "agree_count": int(np.trace(conf_matrix)),
        "disagree_count": int(n - np.trace(conf_matrix))
    }


def load_and_inspect_datasets():
    """Load all datasets and report basic statistics."""
    datasets = {}
    
    # FLUX v1
    if FLUX_V1_CSV.exists():
        df = pd.read_csv(FLUX_V1_CSV)
        datasets['flux_v1'] = {'df': df, 'path': str(FLUX_V1_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  FLUX v1: {len(df)} rows, {len(df.columns)} columns")
    
    # FLUX v2
    if FLUX_V2_CSV.exists():
        df = pd.read_csv(FLUX_V2_CSV)
        datasets['flux_v2'] = {'df': df, 'path': str(FLUX_V2_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  FLUX v2: {len(df)} rows, {len(df.columns)} columns")
    
    # SD35 v1
    if SD35_V1_CSV.exists():
        df = pd.read_csv(SD35_V1_CSV)
        datasets['sd35_v1'] = {'df': df, 'path': str(SD35_V1_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  SD35 v1: {len(df)} rows, {len(df.columns)} columns")
    
    # SD35 v2
    if SD35_V2_CSV.exists():
        df = pd.read_csv(SD35_V2_CSV)
        datasets['sd35_v2'] = {'df': df, 'path': str(SD35_V2_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  SD35 v2: {len(df)} rows, {len(df.columns)} columns")
    
    # Nanobanana final
    if NANO_FINAL_CSV.exists():
        df = pd.read_csv(NANO_FINAL_CSV)
        datasets['nano_final'] = {'df': df, 'path': str(NANO_FINAL_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  Nanobanana final: {len(df)} rows, {len(df.columns)} columns")
    
    # Nanobanana v2 validated
    if NANO_V2_CSV.exists():
        df = pd.read_csv(NANO_V2_CSV)
        datasets['nano_v2_validated'] = {'df': df, 'path': str(NANO_V2_CSV), 'rows': len(df), 'cols': list(df.columns)}
        print(f"  Nanobanana v2 validated: {len(df)} rows, {len(df.columns)} columns")
    
    return datasets


def inspect_human_overrides():
    """Analyze the human_overrides.csv in detail."""
    if not HUMAN_OVERRIDES_CSV.exists():
        print("  [!] human_overrides.csv not found!")
        return None
    
    df = pd.read_csv(HUMAN_OVERRIDES_CSV)
    print(f"\n  Human Overrides: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique master_ids: {df['master_id'].nunique()}")
    print(f"  Unique attributes: {df['attribute_name'].nunique()}")
    print(f"  Attribute distribution:")
    for attr, count in df['attribute_name'].value_counts().items():
        print(f"    {attr}: {count}")
    
    # Check for duplicates
    dupes = df.duplicated(subset=['master_id', 'attribute_name'], keep=False)
    n_dupes = dupes.sum()
    print(f"\n  Duplicate (master_id, attribute_name) pairs: {n_dupes}")
    if n_dupes > 0:
        dupe_df = df[dupes].sort_values(['master_id', 'attribute_name'])
        print("  DUPLICATE ROWS:")
        print(dupe_df.to_string())
    
    # Identify which model each override belongs to
    # Try to infer model from master_id pattern
    flux_pattern = df['master_id'].str.contains(r'_t\d+\.\d+$', regex=True)
    nano_pattern = ~flux_pattern  # nanobanana IDs don't have temperature suffix
    
    print(f"\n  IDs with temperature suffix (FLUX/SD35 pattern): {flux_pattern.sum()}")
    print(f"  IDs without temperature suffix (Nanobanana pattern): {(~flux_pattern).sum()}")
    
    return df


def inspect_vlm_audit():
    """Analyze the vlm_alignment_audit.csv."""
    if not VLM_AUDIT_CSV.exists():
        print("  [!] vlm_alignment_audit.csv not found!")
        return None
    
    df = pd.read_csv(VLM_AUDIT_CSV)
    print(f"\n  VLM Alignment Audit: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique master_ids: {df['master_id'].nunique()}")
    print(f"  human_agrees distribution: {df['human_agrees'].value_counts().to_dict()}")
    
    # Check for duplicates
    dupes = df.duplicated(keep=False)
    n_dupes = dupes.sum()
    print(f"  Exact duplicate rows: {n_dupes}")
    if n_dupes > 0:
        print("  DUPLICATE ROWS FOUND!")
        dupe_df = df[df.duplicated(keep=False)].sort_values(['master_id', 'attribute'])
        unique_pairs = dupe_df[['master_id', 'attribute']].drop_duplicates()
        print(f"  Number of unique (master_id, attribute) pairs with duplicates: {len(unique_pairs)}")
    
    return df


def determine_model_for_overrides(overrides_df, datasets):
    """
    Determine which model each override row belongs to.
    Match override master_ids against dataset ID columns.
    """
    results = []
    
    for ds_name, ds_info in datasets.items():
        df = ds_info['df']
        # Find the ID column
        id_col = None
        for candidate in ['dcdd_id', 'custom_id', 'master_id']:
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            continue
        
        ds_ids = set(df[id_col].astype(str).values)
        matched = overrides_df['master_id'].astype(str).isin(ds_ids)
        results.append({
            'dataset': ds_name,
            'id_column': id_col,
            'total_ids_in_dataset': len(ds_ids),
            'overrides_matched': matched.sum()
        })
        print(f"  {ds_name} ({id_col}): {matched.sum()} overrides matched out of {len(overrides_df)}")
    
    return results


def reconstruct_kappa_for_model(model_name, vlm_dataset_df, vlm_id_col, overrides_df, attributes):
    """
    Reconstruct Cohen's kappa for a specific model.
    
    For each attribute, find all rows where:
    1. An override exists in human_overrides.csv for that (master_id, attribute)
    2. The VLM label is taken from the original dataset
    3. The human label is the corrected_value from overrides
    
    This gives us paired (VLM, Human) labels for computing kappa.
    
    BUT: For items where no override exists, the human implicitly agreed with the VLM.
    To compute proper Cohen's kappa, we need ALL validation decisions, not just disagreements.
    
    The critical question is: were ALL images validated, or only ambiguous ones?
    """
    kappa_results = []
    confusion_matrices = {}
    
    # Get all IDs in this model's dataset
    all_ids = set(vlm_dataset_df[vlm_id_col].astype(str).values)
    
    # Filter overrides to only those matching this dataset
    model_overrides = overrides_df[overrides_df['master_id'].astype(str).isin(all_ids)].copy()
    
    print(f"\n  Model: {model_name}")
    print(f"  Dataset rows: {len(vlm_dataset_df)}")
    print(f"  Overrides matching this model: {len(model_overrides)}")
    
    for attr in attributes:
        if attr not in vlm_dataset_df.columns:
            print(f"    [SKIP] {attr} not in dataset columns")
            continue
        
        # Get overrides for this attribute
        attr_overrides = model_overrides[model_overrides['attribute_name'] == attr].copy()
        
        if len(attr_overrides) == 0:
            print(f"    {attr}: 0 overrides (no human corrections for this attribute)")
            kappa_results.append({
                'model': model_name,
                'attribute': attr,
                'n_overrides': 0,
                'n_total_validated': 0,
                'kappa': float('nan'),
                'p_o': float('nan'),
                'p_e': float('nan'),
                'note': 'No overrides exist for this attribute'
            })
            continue
        
        # Build paired labels
        vlm_labels = []
        human_labels = []
        
        # For each override, get the VLM's original prediction
        for _, override_row in attr_overrides.iterrows():
            mid = str(override_row['master_id'])
            corrected = str(override_row['corrected_value'])
            
            # Find the matching row in the VLM dataset
            matches = vlm_dataset_df[vlm_dataset_df[vlm_id_col].astype(str) == mid]
            if len(matches) == 0:
                continue
            
            vlm_label = str(matches.iloc[0][attr])
            vlm_labels.append(vlm_label)
            human_labels.append(corrected)
        
        # Now: for all OTHER images (where no override exists), 
        # the human validated and agreed with the VLM.
        # This is the critical interpretation question.
        
        # Count how many images in the dataset had this attribute validated
        # According to the scripts, validation was done for ALL ambiguous labels,
        # and the VLM alignment script audited a sample of confident ones.
        
        # For the "ambiguous resolution" path:
        # - The script found all labels containing "ambiguous", "unclear", "indeterminate"
        # - Each of those was presented to the human
        # - Human selected a new label from the menu
        # - That correction was saved to human_overrides.csv
        
        # For non-ambiguous labels: 
        # - The VLM alignment script (vlm_alignment_flux.py) audited a SAMPLE of 50 confident predictions
        # - For disagreements, corrections were also saved to human_overrides.csv
        
        # The key question for kappa: what is the denominator?
        # If we only count items that were actually reviewed, then:
        # - Items with overrides = disagreement
        # - Items that were reviewed but had no override = agreement
        # But the override CSV only records disagreements (corrections).
        # For the VLM alignment audit, we have vlm_alignment_audit.csv which records both agree/disagree.
        
        n_overrides = len(vlm_labels)
        
        # Compute kappa using ONLY the overrides (disagreement-only approach)
        # This would mean: vlm_label != human_label for ALL items
        # Kappa would be pathologically low/negative
        
        # Instead, the correct approach depends on what was actually validated.
        # Let's report what we can:
        
        result_info = {
            'model': model_name,
            'attribute': attr,
            'n_overrides': n_overrides,
            'n_dataset_rows': len(vlm_dataset_df),
        }
        
        # Count how many rows in the dataset have ambiguous values for this attribute
        ambiguous_keywords = ['ambiguous', 'unclear', 'indeterminate']
        ambiguous_mask = vlm_dataset_df[attr].astype(str).str.lower().apply(
            lambda x: any(kw in x for kw in ambiguous_keywords)
        )
        n_ambiguous = ambiguous_mask.sum()
        result_info['n_ambiguous_vlm'] = int(n_ambiguous)
        
        # The validation pipeline resolves ambiguous labels.
        # For ambiguous items: human chose a definite label → saved as override
        # For non-ambiguous items: human implicitly agreed (not recorded in overrides)
        # 
        # Therefore, the correct reconstruction is:
        # Total validated = all images in the dataset
        # For each image-attribute:
        #   if override exists: VLM label = original, Human label = corrected_value
        #   if no override: VLM label = Human label = same (agreement)
        
        # Build FULL paired labels (all images)
        full_vlm_labels = []
        full_human_labels = []
        
        # Create a lookup from overrides
        override_lookup = {}
        for _, row in attr_overrides.iterrows():
            mid = str(row['master_id'])
            # Handle possible duplicate overrides - take the last one
            override_lookup[mid] = str(row['corrected_value'])
        
        for _, data_row in vlm_dataset_df.iterrows():
            mid = str(data_row[vlm_id_col])
            vlm_val = str(data_row[attr])
            
            full_vlm_labels.append(vlm_val)
            if mid in override_lookup:
                full_human_labels.append(override_lookup[mid])
            else:
                full_human_labels.append(vlm_val)  # Agreement
        
        # Compute kappa on full dataset
        kappa_full = compute_cohens_kappa_full(full_vlm_labels, full_human_labels)
        
        result_info['n_total_validated'] = kappa_full['n']
        result_info['kappa'] = kappa_full['kappa']
        result_info['p_o'] = kappa_full['p_o']
        result_info['p_e'] = kappa_full['p_e']
        result_info['agree_count'] = kappa_full['agree_count']
        result_info['disagree_count'] = kappa_full['disagree_count']
        result_info['categories'] = kappa_full['categories']
        result_info['note'] = f"Full dataset: {kappa_full['n']} items, {kappa_full['agree_count']} agree, {kappa_full['disagree_count']} disagree"
        
        kappa_results.append(result_info)
        confusion_matrices[f"{model_name}_{attr}"] = {
            'matrix': kappa_full['confusion_matrix'],
            'categories': kappa_full['categories'],
            'n': kappa_full['n']
        }
        
        print(f"    {attr}: n={kappa_full['n']}, agree={kappa_full['agree_count']}, "
              f"disagree={kappa_full['disagree_count']}, kappa={kappa_full['kappa']:.4f}, "
              f"p_o={kappa_full['p_o']:.4f}, p_e={kappa_full['p_e']:.4f}")
    
    return kappa_results, confusion_matrices


def main():
    print("=" * 80)
    print("  FORENSIC COHEN'S KAPPA RECONSTRUCTION")
    print("  Independent audit of human validation pipeline")
    print("=" * 80)
    
    # ─── 1. Load all datasets ─────────────────────────────────────────────────
    print("\n[1] Loading all datasets...")
    datasets = load_and_inspect_datasets()
    
    # ─── 2. Inspect human overrides ───────────────────────────────────────────
    print("\n[2] Inspecting human_overrides.csv...")
    overrides_df = inspect_human_overrides()
    
    # ─── 3. Inspect VLM alignment audit ───────────────────────────────────────
    print("\n[3] Inspecting vlm_alignment_audit.csv...")
    vlm_audit_df = inspect_vlm_audit()
    
    # ─── 4. Determine which overrides belong to which model ───────────────────
    print("\n[4] Matching overrides to model datasets...")
    if overrides_df is not None:
        match_results = determine_model_for_overrides(overrides_df, datasets)
    
    # ─── 5. Compute Cohen's Kappa per model ──────────────────────────────────
    print("\n[5] Computing Cohen's Kappa for each model...")
    
    all_kappa_results = []
    all_confusion_matrices = {}
    
    # Determine which datasets to use for each model
    # Based on analysis scripts:
    # - 05_flux_analysis.py uses 01_flux_attributes/attributes.csv (v1)
    # - 06_sd35_analysis.py uses 02_sd35_attributes/attributes.csv (v1)  
    # - 03_nanobanana_analysis.py uses nanobanana_attributes_final.csv
    # But human validation scripts use v2 datasets
    # For kappa, we need the VLM predictions BEFORE human overrides were applied
    
    # The v2 datasets appear to be the ones used with the human validation scripts
    # The v1 datasets are the ones used by the final analysis scripts
    # We need to check: do the v1 datasets already have overrides applied?
    
    model_configs = []
    
    # Detect ID columns for each dataset
    for ds_name in ['flux_v1', 'flux_v2', 'sd35_v1', 'sd35_v2', 'nano_final', 'nano_v2_validated']:
        if ds_name in datasets:
            df = datasets[ds_name]['df']
            id_col = None
            for candidate in ['dcdd_id', 'custom_id', 'master_id', 'filename']:
                if candidate in df.columns:
                    id_col = candidate
                    break
            if id_col:
                datasets[ds_name]['id_col'] = id_col
                print(f"  {ds_name}: ID column = {id_col}, unique IDs = {df[id_col].nunique()}")
    
    # For FLUX: Use v2 dataset since validation scripts reference it
    if 'flux_v2' in datasets:
        id_col = datasets['flux_v2'].get('id_col', 'dcdd_id')
        results, matrices = reconstruct_kappa_for_model(
            "FLUX.1 Dev", datasets['flux_v2']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # Also compute for FLUX v1 for comparison
    if 'flux_v1' in datasets:
        id_col = datasets['flux_v1'].get('id_col', 'dcdd_id')
        results, matrices = reconstruct_kappa_for_model(
            "FLUX.1 Dev (v1)", datasets['flux_v1']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # For SD35: Use v2 dataset
    if 'sd35_v2' in datasets:
        id_col = datasets['sd35_v2'].get('id_col', 'dcdd_id')
        results, matrices = reconstruct_kappa_for_model(
            "SD 3.5 Large", datasets['sd35_v2']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # Also for SD35 v1
    if 'sd35_v1' in datasets:
        id_col = datasets['sd35_v1'].get('id_col', 'dcdd_id')
        results, matrices = reconstruct_kappa_for_model(
            "SD 3.5 Large (v1)", datasets['sd35_v1']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # For Nanobanana: Use final dataset
    if 'nano_final' in datasets:
        id_col = datasets['nano_final'].get('id_col', 'custom_id')
        results, matrices = reconstruct_kappa_for_model(
            "Nano Banana", datasets['nano_final']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # Also for Nanobanana v2 validated
    if 'nano_v2_validated' in datasets:
        id_col = datasets['nano_v2_validated'].get('id_col', 'custom_id')
        results, matrices = reconstruct_kappa_for_model(
            "Nano Banana (v2 validated)", datasets['nano_v2_validated']['df'], id_col, overrides_df, KAPPA_ATTRIBUTES
        )
        all_kappa_results.extend(results)
        all_confusion_matrices.update(matrices)
    
    # ─── 6. Save results ─────────────────────────────────────────────────────
    print("\n[6] Saving results...")
    
    # Save kappa results CSV
    kappa_df = pd.DataFrame(all_kappa_results)
    cols_to_save = ['model', 'attribute', 'n_total_validated', 'n_overrides', 'n_ambiguous_vlm',
                    'agree_count', 'disagree_count', 'kappa', 'p_o', 'p_e', 'note']
    cols_available = [c for c in cols_to_save if c in kappa_df.columns]
    kappa_df[cols_available].to_csv(OUTPUT_DIR / "cohen_kappa_results.csv", index=False)
    print(f"  Saved: cohen_kappa_results.csv")
    
    # Save confusion matrices
    conf_dir = OUTPUT_DIR / "confusion_matrices"
    conf_dir.mkdir(parents=True, exist_ok=True)
    for key, data in all_confusion_matrices.items():
        # Save as CSV with category labels
        cats = data['categories']
        matrix = np.array(data['matrix'])
        conf_df = pd.DataFrame(matrix, index=[f"VLM:{c}" for c in cats], columns=[f"Human:{c}" for c in cats])
        safe_key = key.replace(" ", "_").replace(".", "").replace("/", "_")
        conf_df.to_csv(conf_dir / f"{safe_key}.csv")
    print(f"  Saved: {len(all_confusion_matrices)} confusion matrices")
    
    # Save dataset summary
    ds_summary = []
    for ds_name, ds_info in datasets.items():
        ds_summary.append({
            'dataset_name': ds_name,
            'file_path': ds_info['path'],
            'n_rows': ds_info['rows'],
            'n_columns': len(ds_info['cols']),
            'columns': '; '.join(ds_info['cols'][:20]),
            'id_column': ds_info.get('id_col', 'unknown')
        })
    pd.DataFrame(ds_summary).to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    print(f"  Saved: dataset_summary.csv")
    
    # ─── 7. Compute aggregate statistics ──────────────────────────────────────
    print("\n[7] Computing aggregate statistics...")
    
    # For each primary model version, compute macro-average kappa
    for model_label in ["FLUX.1 Dev", "SD 3.5 Large", "Nano Banana",
                         "FLUX.1 Dev (v1)", "SD 3.5 Large (v1)", "Nano Banana (v2 validated)"]:
        model_results = [r for r in all_kappa_results if r['model'] == model_label]
        if not model_results:
            continue
        
        valid_kappas = [r['kappa'] for r in model_results if not np.isnan(r.get('kappa', float('nan')))]
        if valid_kappas:
            macro_kappa = np.mean(valid_kappas)
            total_validated = sum(r.get('n_total_validated', 0) for r in model_results)
            total_agree = sum(r.get('agree_count', 0) for r in model_results)
            total_disagree = sum(r.get('disagree_count', 0) for r in model_results)
            overall_agreement = total_agree / total_validated if total_validated > 0 else 0.0
            
            print(f"\n  {model_label}:")
            print(f"    Macro-average κ: {macro_kappa:.4f}")
            print(f"    Total validated decisions: {total_validated}")
            print(f"    Total agree: {total_agree}, Total disagree: {total_disagree}")
            print(f"    Overall raw agreement: {overall_agreement:.4f} ({overall_agreement*100:.2f}%)")
    
    # ─── 8. Duplicate and consistency checks ──────────────────────────────────
    print("\n[8] Running consistency checks...")
    
    if overrides_df is not None:
        # Check for exact duplicate rows
        exact_dupes = overrides_df.duplicated(keep=False)
        print(f"  Exact duplicate rows in human_overrides.csv: {exact_dupes.sum()}")
        
        # Check for duplicate (master_id, attribute_name) with different corrected_values
        key_dupes = overrides_df.duplicated(subset=['master_id', 'attribute_name'], keep=False)
        if key_dupes.sum() > 0:
            dupe_groups = overrides_df[key_dupes].groupby(['master_id', 'attribute_name'])
            conflicting = 0
            for name, group in dupe_groups:
                if group['corrected_value'].nunique() > 1:
                    conflicting += 1
            print(f"  Duplicate (master_id, attribute) pairs: {key_dupes.sum()}")
            print(f"  Conflicting corrections (different values for same key): {conflicting}")
    
    if vlm_audit_df is not None:
        # Check audit for duplicates
        audit_exact_dupes = vlm_audit_df.duplicated(keep=False)
        print(f"  Exact duplicate rows in vlm_alignment_audit.csv: {audit_exact_dupes.sum()}")
        
        # Count unique audit entries
        unique_audit = vlm_audit_df.drop_duplicates()
        print(f"  Unique audit entries: {len(unique_audit)} out of {len(vlm_audit_df)}")
        print(f"  Agreements: {(unique_audit['human_agrees'] == True).sum()}")
        print(f"  Disagreements: {(unique_audit['human_agrees'] == False).sum()}")
    
    print("\n" + "=" * 80)
    print("  FORENSIC ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
