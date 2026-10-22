#!/usr/bin/env python3
"""
Reproduction script for:
Disability-Conditioned Demographic Drift: Intersectional Stereotyping and
Algorithmic Erasure in Text-to-Image Generative Models (WACV 2027)

Reproduces:
  1. Inter-annotator reliability metrics (Fleiss' kappa, 3 raters, N=300).
  2. Demographic drift and conditional flip rates across evaluated models.
  3. Safety filter refusal rates (algorithmic erasure).
  4. Publication summary charts and tables.
"""

import os
import sys
import argparse
import hashlib
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.inter_rater import fleiss_kappa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEFAULT_REPO_ID = "Red-Pathfinder/dcdd-wacv-2027"
DEFAULT_REVISION = "cf80c456b1e094daeb6fd6a886477b42fe911b5f"
DEFAULT_CACHE_DIR = "dataset_cache"
DEFAULT_OUTPUT_DIR = "results/reproduction"

VISUAL_ATTRIBUTES = [
    "gender_presentation", "perceived_race", "skin_tone", "perceived_age",
    "sentiment", "environment_type", "professional_attire", "gaze_direction",
    "physical_framing", "activity_engagement", "social_integration", "image_style"
]

RELIABILITY_ATTRIBUTES = [
    "gender_presentation", "social_integration", "activity_engagement",
    "skin_tone", "physical_framing", "sentiment"
]

TABULAR_PATTERNS = [
    "metadata/*",
    "human_reliability_study/*.csv",
    "human_reliability_study/README.md",
    "README.md",
    "checksums.sha256"
]


def acquire_dataset(data_dir: Optional[Path], repo_id: str, revision: str, mode: str, force_download: bool) -> Path:
    """Locate local dataset or download pinned version from Hugging Face Hub."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if data_dir is not None:
        target = data_dir.resolve()
        if (target / "metadata" / "dataset_manifest.csv").exists():
            print(f"Using local dataset directory: {target}")
            return target
        print(f"Specified directory '{target}' does not contain dataset_manifest.csv. Downloading.")
    else:
        target = (project_root / DEFAULT_CACHE_DIR).resolve()

    manifest_file = target / "metadata" / "dataset_manifest.csv"
    if not force_download and manifest_file.exists():
        if mode == "tabular":
            print(f"Using cached tabular dataset: {target}")
            return target
        elif mode == "full" and (target / "data" / "flux").exists():
            print(f"Using cached full dataset: {target}")
            return target

    print(f"Downloading dataset from Hugging Face ({repo_id} @ {revision[:7]})...")
    try:
        from huggingface_hub import snapshot_download
        allow_patterns = TABULAR_PATTERNS if mode == "tabular" else None

        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.git*", ".gitattributes"]
        )
        return Path(downloaded_path)

    except Exception as e:
        err = str(e)
        print(f"\nError downloading dataset from Hugging Face Hub:\n{err}", file=sys.stderr)
        if any(keyword in err.lower() for keyword in ["401", "403", "gated", "not found", "unauthorized"]):
            print(
                "\nNote: The dataset repository is currently private.\n"
                "Please configure access using either:\n"
                "  export HF_TOKEN=\"<your_token>\"\n"
                "  or: hf auth login\n",
                file=sys.stderr
            )
        sys.exit(1)


def validate_dataset(dataset_dir: Path, mode: str) -> Dict[str, pd.DataFrame]:
    """Verify integrity of dataset tables and checksums."""
    metadata_dir = dataset_dir / "metadata"
    required = ["dataset_manifest.csv", "annotations.csv", "generation_metadata.csv", "prompts.csv"]
    for fname in required:
        fpath = metadata_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required dataset file: {fpath}")

    df_manifest = pd.read_csv(metadata_dir / "dataset_manifest.csv")
    df_annotations = pd.read_csv(metadata_dir / "annotations.csv")
    df_gen_meta = pd.read_csv(metadata_dir / "generation_metadata.csv")
    df_prompts = pd.read_csv(metadata_dir / "prompts.csv")

    total_attempts = len(df_manifest)
    completed_images = int((df_manifest["has_image"] == True).sum())
    refusals = int((df_manifest["attempt_status"] == "safety_refusal").sum())

    assert total_attempts == 2690, f"Expected 2690 attempts, found {total_attempts}"
    assert completed_images == 2472, f"Expected 2472 completed images, found {completed_images}"
    assert refusals == 218, f"Expected 218 safety refusals, found {refusals}"

    # Verify checksums for present files
    checksum_file = dataset_dir / "checksums.sha256"
    if checksum_file.exists():
        verified = 0
        with open(checksum_file, "r", encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                expected_sha, rel_path = line.split(maxsplit=1)
                target_path = dataset_dir / rel_path

                # README.md is documentation that evolves independently of data assets
                if rel_path == "README.md":
                    continue

                if mode == "tabular" and not target_path.exists():
                    continue

                if not target_path.exists():
                    raise FileNotFoundError(f"Missing file listed in checksums: {rel_path}")

                with open(target_path, "rb") as f:
                    actual_sha = hashlib.sha256(f.read()).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(f"Checksum mismatch for {rel_path}")
                verified += 1

        print(f"Verified {verified} file checksums.")

    if mode == "full":
        for _, row in df_manifest[df_manifest["has_image"] == True].iterrows():
            img_path = dataset_dir / row["image_filename"]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image file: {row['image_filename']}")
        print("Verified all 2,472 image binaries on disk.")

    return {
        "manifest": df_manifest,
        "annotations": df_annotations,
        "generation_metadata": df_gen_meta,
        "prompts": df_prompts,
    }


def reproduce_reliability_study(dataset_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Compute 3-annotator agreement metrics across 300 blinded stimulus images."""
    study_dir = dataset_dir / "human_reliability_study"
    annotator_files = [study_dir / f"annotator_{i}.csv" for i in (1, 2, 3)]
    if not all(p.exists() for p in annotator_files):
        print("Reliability study files not found, skipping inter-annotator analysis.")
        return pd.DataFrame()

    a1 = pd.read_csv(annotator_files[0]).set_index("image_id")
    a2 = pd.read_csv(annotator_files[1]).set_index("image_id")
    a3 = pd.read_csv(annotator_files[2]).set_index("image_id")

    shared = a1.index.intersection(a2.index).intersection(a3.index)
    records = []

    print("\nInter-Annotator Agreement (N=300 blinded images, 3 raters):")
    print(f"{'Attribute':<22s} {'Fleiss κ':>10s} {'Paper κ':>10s} {'Pairwise %':>12s}")
    print("-" * 58)

    canonical_kappa = {
        "gender_presentation": 0.8205,
        "social_integration": 0.8142,
        "activity_engagement": 0.8147,
        "sentiment": 0.6995,
        "skin_tone": 0.6427,
        "physical_framing": 0.5761
    }

    for attr in RELIABILITY_ATTRIBUTES:
        s1, s2, s3 = a1.loc[shared, attr], a2.loc[shared, attr], a3.loc[shared, attr]
        pairwise = ((s1 == s2).mean() + (s1 == s3).mean() + (s2 == s3).mean()) / 3.0 * 100

        cats = sorted(list(set(s1).union(set(s2)).union(set(s3))))
        cat_map = {c: i for i, c in enumerate(cats)}
        matrix = np.zeros((len(shared), len(cats)), dtype=int)
        for i in range(len(shared)):
            matrix[i, cat_map[s1.iloc[i]]] += 1
            matrix[i, cat_map[s2.iloc[i]]] += 1
            matrix[i, cat_map[s3.iloc[i]]] += 1

        k_val = fleiss_kappa(matrix)
        paper_k = canonical_kappa.get(attr, np.nan)
        print(f"{attr:<22s} {k_val:>10.4f} {paper_k:>10.4f} {pairwise:>11.1f}%")

        records.append({
            "attribute": attr,
            "fleiss_kappa": round(k_val, 4),
            "paper_kappa": paper_k,
            "mean_pairwise_agreement_pct": round(pairwise, 2),
            "n_images": len(shared)
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_dir / "table_reliability_metrics.csv", index=False)
    return out_df


def calc_cramers_v(contingency_table: np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013)."""
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    if n <= 1 or min(r, c) <= 1:
        return 0.0
    phi2 = max(0.0, (chi2 / n) - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def calc_cfr(df: pd.DataFrame, attribute: str) -> float:
    """Conditional Flip Rate (CFR %) between neutral and conditioned prompt pairs."""
    neutral_df = df[df["image_type"] == "neutral"].set_index("pair_key")
    cond_df = df[df["image_type"] == "conditioned"]

    flips, total = 0, 0
    for _, row in cond_df.iterrows():
        pair_id = row["pair_key"]
        if pair_id in neutral_df.index:
            n_val = str(neutral_df.loc[[pair_id]].iloc[0][attribute]).lower().strip()
            c_val = str(row[attribute]).lower().strip()
            if n_val not in ["ambiguous", "nan", ""] and c_val not in ["ambiguous", "nan", ""]:
                if n_val != c_val:
                    flips += 1
                total += 1
    return (flips / total * 100.0) if total > 0 else 0.0


def reproduce_drift_analysis(df_manifest: pd.DataFrame, df_annotations: pd.DataFrame, output_dir: Path):
    """Compute demographic drift (Cramér's V), Chi-square independence, and CFR."""
    merged = df_manifest.merge(df_annotations, on=["sample_id", "model"], how="inner")
    merged["pair_key"] = merged["pair_id"].astype(str)

    summaries = []
    models = ["FLUX.1 Dev", "SD 3.5 Large", "Nano Banana 2"]

    for model_name in models:
        m_df = merged[(merged["model"] == model_name) & (merged["attempt_status"] == "completed")]
        if m_df.empty:
            continue

        records = []
        raw_p_values = []

        for attr in VISUAL_ATTRIBUTES:
            if attr not in m_df.columns:
                continue
            sub = m_df.dropna(subset=[attr, "image_type"])
            if sub.empty or sub[attr].nunique() < 2:
                continue
            cross = pd.crosstab(sub["image_type"], sub[attr])
            if cross.shape[0] < 2 or cross.shape[1] < 2:
                continue

            chi2, p_val, dof, _ = chi2_contingency(cross.values, correction=False)
            v_val = calc_cramers_v(cross.values)
            cfr_val = calc_cfr(m_df, attr)

            raw_p_values.append(p_val)
            records.append({
                "model": model_name,
                "attribute": attr,
                "cramers_v": round(v_val, 4),
                "cfr_percent": round(cfr_val, 2),
                "chi2": round(chi2, 3),
                "p_raw": p_val
            })

        if not records:
            continue

        res_df = pd.DataFrame(records)
        _, fdr_p, _, _ = multipletests(raw_p_values, alpha=0.05, method="fdr_bh")
        res_df["p_fdr"] = fdr_p
        res_df["fdr_sig"] = res_df["p_fdr"] < 0.05
        res_df = res_df.sort_values(by="cramers_v", ascending=False).reset_index(drop=True)
        summaries.append(res_df)

        print(f"\n{model_name} Demographic Drift (N={len(m_df)}):")
        print(f"{'Attribute':<22s} {'Cramér V':>10s} {'CFR (%)':>10s} {'FDR Sig':>10s}")
        print("-" * 55)
        for _, r in res_df.head(5).iterrows():
            print(f"{r['attribute']:<22s} {r['cramers_v']:>10.4f} {r['cfr_percent']:>9.1f}% {str(r['fdr_sig']):>10s}")

        slug = model_name.lower().replace(" ", "").replace(".", "")
        res_df.to_csv(output_dir / f"{slug}_statistical_summary.csv", index=False)

    if summaries:
        master_df = pd.concat(summaries, ignore_index=True)
        master_df.to_csv(output_dir / "DCDD_Master_Reproduction_Summary.csv", index=False)

    return merged


def reproduce_erasure_analysis(df_manifest: pd.DataFrame, output_dir: Path):
    """Analyze Google Gemini safety filter refusals across conditions."""
    nano = df_manifest[df_manifest["model"] == "Nano Banana 2"].copy()
    total_refusals = (nano["attempt_status"] == "safety_refusal").sum()

    conditions = ["None (Neutral)", "mobility impairment", "vision impairment", "hearing impairment", "mental health condition"]
    rows = []

    print(f"\nAlgorithmic Erasure / Gemini Safety Refusals (Total Refusals = {total_refusals} / 1925 = {total_refusals/1925*100:.1f}%):")
    print(f"{'Condition':<25s} {'Attempts':>10s} {'Refusals':>10s} {'Refusal %':>12s} {'Share':>10s}")
    print("-" * 70)

    for cond in conditions:
        if "Neutral" in cond:
            c_df = nano[nano["image_type"] == "neutral"]
            name = "neutral"
        else:
            c_df = nano[nano["disability"] == cond]
            name = cond

        c_total = len(c_df)
        c_ref = int((c_df["attempt_status"] == "safety_refusal").sum())
        ref_pct = (c_ref / c_total * 100.0) if c_total > 0 else 0.0
        share_pct = (c_ref / total_refusals * 100.0) if total_refusals > 0 else 0.0

        print(f"{name:<25s} {c_total:>10d} {c_ref:>10d} {ref_pct:>11.1f}% {share_pct:>9.1f}%")
        rows.append({
            "condition": name,
            "attempts": c_total,
            "refusals": c_ref,
            "refusal_rate_pct": round(ref_pct, 2),
            "share_of_all_refusals_pct": round(share_pct, 2)
        })

    erasure_df = pd.DataFrame(rows)
    erasure_df.to_csv(output_dir / "table_erasure_refusals.csv", index=False)
    return erasure_df


def generate_reproduction_figures(merged_df: pd.DataFrame, output_dir: Path):
    """Generate distribution comparison charts."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    flux = merged_df[(merged_df["model"] == "FLUX.1 Dev") & (merged_df["attempt_status"] == "completed")]
    if not flux.empty and "physical_framing" in flux.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
        cross = pd.crosstab(flux["image_type"], flux["physical_framing"], normalize="index") * 100
        cross.plot(kind="bar", ax=ax, colormap="tab10")
        ax.set_title("FLUX.1 [dev]: Physical Framing Distribution (Cramér's V = 0.240)", fontsize=10)
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xlabel("Condition Type", fontsize=9)
        ax.legend(title="Framing", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig.savefig(charts_dir / "fig_flux_physical_framing_drift.png")
        plt.close(fig)

    sd = merged_df[(merged_df["model"] == "SD 3.5 Large") & (merged_df["attempt_status"] == "completed")]
    if not sd.empty and "sentiment" in sd.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200)
        cross = pd.crosstab(sd["image_type"], sd["sentiment"], normalize="index") * 100
        cross.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_title("Stable Diffusion 3.5: Sentiment Shift (Cramér's V = 0.188)", fontsize=10)
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xlabel("Condition Type", fontsize=9)
        ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig.savefig(charts_dir / "fig_sd35_sentiment_shift.png")
        plt.close(fig)

    nano = merged_df[merged_df["model"] == "Nano Banana 2"]
    if not nano.empty:
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
        rates = nano.groupby("disability")["attempt_status"].apply(lambda x: (x == "safety_refusal").mean() * 100)
        rates.plot(kind="barh", ax=ax, color="#d9534f")
        ax.set_title("NanoBanana 2 (Gemini): Refusal Rate by Condition", fontsize=10)
        ax.set_xlabel("Safety Refusal Rate (%)", fontsize=9)
        ax.set_ylabel("Disability Condition", fontsize=9)
        plt.tight_layout()
        fig.savefig(charts_dir / "fig_nanobanana_safety_refusal_erasure.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reproduce DCDD (WACV 2027) benchmark findings.")
    parser.add_argument("--mode", choices=["tabular", "full"], default="tabular",
                        help="Reproduction mode: 'tabular' (default, metadata & stats, ~1.2 MB) or 'full' (all images, ~1.6 GB)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to local dataset directory (skips download if valid).")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help="Hugging Face dataset repository ID.")
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION,
                        help="Pinned Hugging Face revision hash.")
    parser.add_argument("--download", action="store_true",
                        help="Force fresh download from Hugging Face.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out = Path(args.output_dir)
    output_dir = out.resolve() if out.is_absolute() else (project_root / out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    dataset_path = acquire_dataset(data_dir, args.repo_id, args.revision, args.mode, args.download)

    dataset_dict = validate_dataset(dataset_path, args.mode)
    reproduce_reliability_study(dataset_path, output_dir)
    merged_df = reproduce_drift_analysis(dataset_dict["manifest"], dataset_dict["annotations"], output_dir)
    reproduce_erasure_analysis(dataset_dict["manifest"], output_dir)
    generate_reproduction_figures(merged_df, output_dir)

    print(f"\nReproduction complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
