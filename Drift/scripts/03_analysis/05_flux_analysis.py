"""
DCDD Unified Analysis Pipeline — FLUX
=====================================
This is the master script for our EMNLP paper. 
It proves whether FLUX is biased by checking if adding a "disability" 
keyword completely changes the demographics or context of the generated image.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from statsmodels.stats.multitest import multipletests

# Hide annoying warnings and make sure our PDF charts have editable text
warnings.filterwarnings("ignore")
matplotlib.rcParams["pdf.fonttype"] = 42   
matplotlib.rcParams["ps.fonttype"]  = 42

# ── Setup & Constants ─────────────────────────────────────────────────────────
# Script location: C:\Study\Coding\Research\Drift\scripts\03_analysis
# BASE_DIR resolves to: C:\Study\Coding\Research
BASE_DIR = Path(__file__).resolve().parents[3]

# Inputs (Verify the exact name of your flux_attributes folder if you numbered it)
# Inputs
DATA_PATH = BASE_DIR / "Drift" / "data_processed" / "01_flux_attributes" / "attributes.csv"

# Outputs
OUT_DIR   = BASE_DIR / "results" / "charts" / "flux charts"
STATS_CSV = BASE_DIR / "results" / "statistical summary" / "flux_statistical_summary.csv"

DPI           = 300
CONDITION_COL = "image_type"
NEUTRAL_LABEL = "neutral"
COND_LABEL    = "conditioned"

# The 12 specific things we are looking for in every image
VISUAL_ATTRIBUTES = [
    "gender_presentation", "perceived_race", "skin_tone", "perceived_age",
    "sentiment", "environment_type", "professional_attire", "gaze_direction",
    "physical_framing", "activity_engagement", "social_integration", "image_style"
]

# Set up our chart colors (Colorblind safe so reviewers don't complain)
PALETTE = sns.color_palette("colorblind")
NEUTRAL_COLOR = PALETTE[0]   # Blue for baseline
COND_COLOR    = PALETTE[1]   # Orange for when disability is added

# Standard font sizes for the research paper
TITLE_SIZE  = 10
LABEL_SIZE  = 9
TICK_SIZE   = 8
LEGEND_SIZE = 8
ANNOT_SIZE  = 7


# ── Step 1: Loading the Data ──────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Loads our CSV, filters OOD age bins, and figures out pair matching."""
    df = pd.read_csv(path)
    
    # Make sure 'neutral' and 'conditioned' are perfectly lowercase
    df[CONDITION_COL] = df[CONDITION_COL].str.strip().str.lower()
    
    # POST-HOC TRUNCATION: Slices the FLUX dataset from 505 to 385 images
    # to maintain a perfect 1:1 comparative baseline with SD 3.5.
    df = df[df['age_bin'] != 'teen (13-17)']
    
    # Combine seed + profession + age to create a unique ID for the Counterfactual pairs.
    if 'pair_key' not in df.columns:
        df['pair_key'] = df['seed'].astype(str) + "_" + df['profession'] + "_" + df['age_bin']
        
    return df


# ── Step 2: The Math & Statistics ─────────────────────────────────────────────

def calculate_flip_rate(df: pd.DataFrame, attribute: str) -> float:
    """Calculates the Identity Erasure rate (CFR)."""
    neutral_df = df[df[CONDITION_COL] == NEUTRAL_LABEL].set_index('pair_key')
    cond_df = df[df[CONDITION_COL] == COND_LABEL]

    flips = 0
    total_pairs = 0

    for _, row in cond_df.iterrows():
        pair_id = row['pair_key']
        
        if pair_id in neutral_df.index:
            neutral_row = neutral_df.loc[[pair_id]]
            n_val = str(neutral_row.iloc[0][attribute]).lower().strip()
            c_val = str(row[attribute]).lower().strip()

            # Skip 'ambiguous' stuff since we only care about hard identity flips
            if n_val != 'ambiguous' and c_val != 'ambiguous':
                if n_val != c_val:
                    flips += 1
                total_pairs += 1

    return (flips / total_pairs * 100) if total_pairs > 0 else 0.0


def cramers_v(contingency_table: np.ndarray) -> float:
    """Calculates Cramér's V (Effect Size)."""
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    phi2 = max(0, (chi2 / n) - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def test_attribute(df: pd.DataFrame, attribute: str) -> dict:
    """Runs the statistical tests to prove our findings aren't random luck."""
    cross = pd.crosstab(df[CONDITION_COL], df[attribute])
    cross = cross.reindex([NEUTRAL_LABEL, COND_LABEL]).dropna(how="all")
    tab = cross.values.astype(float)

    exp = expected_freq(tab)
    n_sparse = (exp < 5).sum()

    if n_sparse > 0:
        test_used = "Fisher/Merged Fallback"
        _, p, _, _ = chi2_contingency(tab, correction=False) 
    else:
        test_used = "Chi-Square"
        _, p, _, _ = chi2_contingency(tab, correction=False)

    v = cramers_v(tab)

    return {
        "attribute": attribute,
        "raw_p": p,
        "cramers_v": v,
        "test_used": test_used,
    }


def run_statistical_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all the math for all 12 attributes and packages it up."""
    records = []
    for attr in VISUAL_ATTRIBUTES:
        res = test_attribute(df, attr)
        res['cfr_percent'] = calculate_flip_rate(df, attr) 
        records.append(res)
        
    results = pd.DataFrame(records)

    # Benjamini-Hochberg FDR correction
    reject, p_corr, _, _ = multipletests(results["raw_p"], alpha=0.05, method="fdr_bh")
    results["fdr_corrected_p"] = p_corr
    results["significant_fdr"] = reject

    def label_effect(v: float) -> str:
        if v < 0.10: return "Negligible"
        if v < 0.20: return "Small"
        if v < 0.40: return "Medium"
        return "Large"

    results["effect_size_label"] = results["cramers_v"].apply(label_effect)
    results = results.sort_values("cramers_v", ascending=False).reset_index(drop=True)

    return results


def export_statistics(results: pd.DataFrame, path: Path) -> None:
    """Saves our results to a clean CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    export_cols = [
        "attribute", "test_used", "raw_p", "fdr_corrected_p", 
        "cramers_v", "effect_size_label", "significant_fdr", "cfr_percent"
    ]
    out = results[export_cols].copy()
    out.columns = [
        "Attribute", "Test Used", "Raw P-Value", "FDR-Corrected P-Value", 
        "Cramers_V", "Effect Size", "Significant (FDR α=0.05)", "Counterfactual Flip Rate (%)"
    ]
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"[✓] FLUX Statistical Summary saved → {path}")


# ── Step 3: Drawing the Charts ────────────────────────────────────────────────

def apply_publication_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "Proportion (%)") -> None:
    """Strips away default styling for publication-ready charts."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def compute_proportions(df: pd.DataFrame, attribute: str, category_order: list = None) -> pd.DataFrame:
    """Converts raw image counts into percentages for a fair comparison."""
    grp = df.groupby([CONDITION_COL, attribute]).size().reset_index(name="count")
    totals = df.groupby(CONDITION_COL).size().reset_index(name="total")
    grp = grp.merge(totals, on=CONDITION_COL)
    grp["proportion"] = 100 * grp["count"] / grp["total"]
    grp.rename(columns={attribute: "category"}, inplace=True)
    
    if category_order:
        grp["category"] = pd.Categorical(grp["category"], categories=category_order, ordered=True)
        grp = grp.sort_values("category")
    return grp


def plot_grouped_bars(ax: plt.Axes, df: pd.DataFrame, attribute: str, category_order: list, title: str, cramers_v_val: float, p_val: float, sig: bool, cfr: float) -> None:
    """Draws the bars and annotates the statistics."""
    prop = compute_proportions(df, attribute, category_order)
    categories = [c for c in category_order if c in prop["category"].values]
    x = np.arange(len(categories))
    width = 0.35

    neutral_vals, cond_vals = [], []
    for cat in categories:
        sub = prop[prop["category"] == cat]
        n_row = sub[sub[CONDITION_COL] == NEUTRAL_LABEL]["proportion"].values
        c_row = sub[sub[CONDITION_COL] == COND_LABEL]["proportion"].values
        neutral_vals.append(n_row[0] if len(n_row) else 0.0)
        cond_vals.append(c_row[0] if len(c_row) else 0.0)

    bars_n = ax.bar(x - width / 2, neutral_vals, width, color=NEUTRAL_COLOR, label="Neutral", alpha=0.9, edgecolor="white", linewidth=0.5)
    bars_c = ax.bar(x + width / 2, cond_vals, width, color=COND_COLOR, label="Conditioned", alpha=0.9, edgecolor="white", linewidth=0.5)

    for bar in itertools.chain(bars_n, bars_c):
        h = bar.get_height()
        if h > 1.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=ANNOT_SIZE, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=22, ha="right", fontsize=TICK_SIZE)

    sig_marker = "*" if sig else "ns"
    p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
    annot = f"V={cramers_v_val:.2f} | CFR={cfr:.1f}%\n{p_str} {sig_marker}"
    ax.text(0.97, 0.95, annot, transform=ax.transAxes, ha="right", va="top", fontsize=ANNOT_SIZE, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.8))
    
    apply_publication_style(ax, title)


# ── The Specific Paper Figures ────────────────────────────────────────────────

def plot_agency_penalty(df: pd.DataFrame, results: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    fig.suptitle("FLUX: The Agency Penalty — Physical Framing & Activity Engagement", fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)
    
    attrs = {
        "physical_framing": {"order": ["Expansive/Commanding", "Neutral", "Diminished/Hunched"], "title": "(a) Physical Framing"},
        "activity_engagement": {"order": ["Active/Working", "Idle/Observing", "Being Assisted"], "title": "(b) Activity Engagement"},
    }
    
    for ax, (attr, cfg) in zip(axes, attrs.items()):
        row = results[results["attribute"] == attr].iloc[0]
        plot_grouped_bars(ax, df, attr, cfg["order"], cfg["title"], row["cramers_v"], row["fdr_corrected_p"], row["significant_fdr"], row["cfr_percent"])
        
    fig.legend(handles=[mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral Baseline"), mpatches.Patch(color=COND_COLOR, label="Disability-Conditioned")], loc="lower center", ncol=2, fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_flux_agency_penalty.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_demographic_drift(df: pd.DataFrame, results: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    fig.suptitle("FLUX: Demographic Drift — Gender Presentation & Perceived Race", fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)
    
    attrs = {
        "gender_presentation": {"order": ["Male", "Female", "Ambiguous"], "title": "(a) Gender Presentation"},
        "perceived_race": {"order": ["White", "Ambiguous", "Asian", "Black", "Multiracial", "Hispanic/Latino"], "title": "(b) Perceived Race"},
    }
    
    for ax, (attr, cfg) in zip(axes, attrs.items()):
        row = results[results["attribute"] == attr].iloc[0]
        present = df[attr].unique().tolist()
        order = [c for c in cfg["order"] if c in present]
        plot_grouped_bars(ax, df, attr, order, cfg["title"], row["cramers_v"], row["fdr_corrected_p"], row["significant_fdr"], row["cfr_percent"])
        
    fig.legend(handles=[mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral Baseline"), mpatches.Patch(color=COND_COLOR, label="Disability-Conditioned")], loc="lower center", ncol=2, fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_flux_demographic_drift.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_socioeconomic_downgrade(df: pd.DataFrame, results: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    fig.suptitle("FLUX: Socioeconomic Downgrade — Professional Attire & Environment", fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)
    
    attrs = {
        "professional_attire": {"order": ["Formal/Uniform", "Business Casual", "Casual/Informal", "Medical/Institutional"], "title": "(a) Professional Attire"},
        "environment_type": {"order": ["Workplace", "Studio/Abstract", "Outdoor/Public", "Domestic", "Clinical/Medical"], "title": "(b) Environment Type"},
    }
    
    for ax, (attr, cfg) in zip(axes, attrs.items()):
        row = results[results["attribute"] == attr].iloc[0]
        present = df[attr].unique().tolist()
        order = [c for c in cfg["order"] if c in present]
        plot_grouped_bars(ax, df, attr, order, cfg["title"], row["cramers_v"], row["fdr_corrected_p"], row["significant_fdr"], row["cfr_percent"])
        
    fig.legend(handles=[mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral Baseline"), mpatches.Patch(color=COND_COLOR, label="Disability-Conditioned")], loc="lower center", ncol=2, fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_flux_socioeconomic_downgrade.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── The Control Center ────────────────────────────────────────────────────────

def main():
    # Make sure output directories exist before writing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="white", palette="colorblind")

    print("=" * 60)
    print("  DCDD Analysis Pipeline — FLUX (EMNLP 2026)")
    print("=" * 60)

    # Note: Double check if your folder is 'flux_attributes' or '01_flux_attributes'
    print(f"\n[1] Loading paired data from: {DATA_PATH.parent.name}/{DATA_PATH.name}")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"[!] CRITICAL ERROR: Could not find {DATA_PATH}")
        return
        
    print("\n[2] Running Macro & Micro statistical tests (with Age-Bin Truncation)...")
    results = run_statistical_pipeline(df)

    print("\n    ── Results Preview ───────────────────────────────────────")
    print(results[["attribute", "cramers_v", "cfr_percent", "significant_fdr"]].to_string(index=False))

    print(f"\n[3] Exporting master statistics...")
    export_statistics(results, STATS_CSV)

    print(f"\n[4] Generating publication-grade figures → {OUT_DIR.name}/")
    plot_agency_penalty(df, results, OUT_DIR)
    plot_demographic_drift(df, results, OUT_DIR)
    plot_socioeconomic_downgrade(df, results, OUT_DIR)

    print("\n" + "=" * 60)
    print("  FLUX Pipeline complete.")
    print(f"  Master Stats CSV : {STATS_CSV}")
    print(f"  Figures          : {OUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()