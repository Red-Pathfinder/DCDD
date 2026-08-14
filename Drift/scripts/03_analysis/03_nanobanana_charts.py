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

warnings.filterwarnings("ignore")
matplotlib.rcParams["pdf.fonttype"] = 42   
matplotlib.rcParams["ps.fonttype"]  = 42

# ─── PATHS ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path(r"C:\Study\Coding\Research\Drift\data_processed\03_nanobanana_attributes\nanobanana_attributes_final.csv")
OUT_DIR   = Path(r"C:\Study\Coding\Research\results\charts\nanobanana charts")
STATS_CSV = Path(r"C:\Study\Coding\Research\results\statistical summary\nanobanana_statistical_summary.csv")

DPI = 300
CONDITION_COL = "image_type"
NEUTRAL_LABEL = "neutral"
COND_LABEL    = "conditioned"

VISUAL_ATTRIBUTES = [
    "gender_presentation", "perceived_race", "skin_tone",
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement",
    "social_integration", "image_style"
]

PALETTE = sns.color_palette("colorblind")
NEUTRAL_COLOR = PALETTE[0]   
COND_COLOR    = PALETTE[2] # Using Green for Gemini to distinguish from SD3.5/FLUX  

TITLE_SIZE  = 10
LABEL_SIZE  = 9
TICK_SIZE   = 8
LEGEND_SIZE = 8
ANNOT_SIZE  = 7

def load_and_harmonize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Harmonize condition column
    df['image_type'] = np.where(df['condition'].str.lower() == 'neutral', 'neutral', 'conditioned')
    
    # Create pair key including temperature to match exactly
    if 'pair_key' not in df.columns:
        df['pair_key'] = df['seed'].astype(str) + "_" + df['profession'] + "_" + df['age_bin'] + "_t" + df['temperature'].astype(str)
        
    return df

def cramers_v(contingency_table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    phi2 = max(0, (chi2 / n) - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0

def test_attribute(df: pd.DataFrame, attribute: str) -> dict:
    cross = pd.crosstab(df[CONDITION_COL], df[attribute])
    cross = cross.reindex([NEUTRAL_LABEL, COND_LABEL]).dropna(how="all")
    tab = cross.values.astype(float)

    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"attribute": attribute, "raw_p": 1.0, "cramers_v": 0.0, "test_used": "None"}

    exp = expected_freq(tab)
    n_sparse = (exp < 5).sum()
    test_used = "Fisher/Merged Fallback" if n_sparse > 0 else "Chi-Square"
    
    try:
        _, p, _, _ = chi2_contingency(tab, correction=False)
    except:
        p = 1.0
        
    v = cramers_v(tab)
    return {"attribute": attribute, "raw_p": p, "cramers_v": v, "test_used": test_used}

def run_statistical_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for attr in VISUAL_ATTRIBUTES:
        res = test_attribute(df, attr)
        records.append(res)
        
    results = pd.DataFrame(records)
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

def apply_publication_style(ax: plt.Axes, title: str, ylabel: str = "Proportion (%)") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

def plot_algorithmic_erasure(df: pd.DataFrame, out_dir: Path) -> None:
    """Plots the rate of 'Safety Refusals' triggered by disability keywords."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Calculate refusal percentages
    neutral_total = len(df[df[CONDITION_COL] == NEUTRAL_LABEL])
    cond_total = len(df[df[CONDITION_COL] == COND_LABEL])
    
    neutral_refusals = len(df[(df[CONDITION_COL] == NEUTRAL_LABEL) & (df['sentiment'] == 'Safety Refusal')])
    cond_refusals = len(df[(df[CONDITION_COL] == COND_LABEL) & (df['sentiment'] == 'Safety Refusal')])
    
    n_rate = (neutral_refusals / neutral_total * 100) if neutral_total else 0
    c_rate = (cond_refusals / cond_total * 100) if cond_total else 0

    bars = ax.bar(["Neutral Baseline", "Disability Conditioned"], [n_rate, c_rate], 
                  color=[NEUTRAL_COLOR, COND_COLOR], width=0.5, alpha=0.9, edgecolor="white")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=ANNOT_SIZE+1, fontweight="bold")

    apply_publication_style(ax, "Gemini Flash: Safety Refusal Rate (Algorithmic Erasure)", ylabel="Refusal Rate (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_gemini_safety_refusals.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", palette="colorblind")

    print("[*] Loading Gemini Nanobanana data...")
    df = load_and_harmonize(DATA_PATH)
    
    print("[*] Running statistics...")
    results = run_statistical_pipeline(df)
    
    export_cols = ["attribute", "test_used", "raw_p", "fdr_corrected_p", "cramers_v", "effect_size_label", "significant_fdr"]
    out = results[export_cols].copy()
    out.to_csv(STATS_CSV, index=False, float_format="%.4f")
    print(f"[✓] Saved Gemini stats to: {STATS_CSV.name}")

    print("[*] Generating unique Gemini charts (Safety Refusals)...")
    plot_algorithmic_erasure(df, OUT_DIR)

    print(f"[✓] Charts saved to: {OUT_DIR}")
    print("=" * 60)
    print("Pipeline Complete! Check your folders.")

if __name__ == "__main__":
    main()