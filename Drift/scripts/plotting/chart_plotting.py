import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

EXCEL_PATH = Path(r"C:\Study\Coding\Research\Drift\scripts\draft_scaled_comparision\DCDD_Final_Master_Analysis.xlsx")
OUTPUT_DIR = Path(r"C:\Study\Coding\Research\results\charts")


def load_sheet(sheet_name: str) -> pd.DataFrame | None:
    if not EXCEL_PATH.exists():
        print(f"[!] Error: Missing required file: {EXCEL_PATH}")
        return None
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)


def plot_gemini_refusal():
    df = load_sheet("sentiment")
    if df is None:
        return

    gemini_df = df[df["Model"].str.contains("Gemini", na=False, case=False)]

    cond_keywords = ["Baseline", "mobility", "vision", "hearing", "mental health"]
    conditions   = ["Neutral", "Mobility", "Vision", "Hearing", "Mental Health"]
    total = 385

    refusals = []
    for kw in cond_keywords:
        match = gemini_df[gemini_df["Condition"].str.contains(kw, na=False, case=False)]
        refusals.append(float(match.iloc[0]["Safety Refusal"]) if not match.empty else 0.0)

    rates  = [(r / total) * 100 for r in refusals]
    colors = ["#888888", "#888888", "#888888", "#888888", "#A32D2D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(conditions, rates, color=colors)
    ax.set_title("Gemini 3.1 Flash — Safety Refusal Rate by Condition",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Refusal Rate (%)")
    ax.set_ylim(0, 45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1,
                f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig1_gemini_refusal.png", dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {OUTPUT_DIR / 'fig1_gemini_refusal.png'}")


def plot_flux_gender_drift():
    df = load_sheet("gender_presentation")
    if df is None:
        return

    flux_df = df[df["Model"].str.contains("FLUX", na=False, case=False)]

    cond_keywords = ["Baseline", "mobility", "vision", "hearing", "mental health"]
    short_names   = ["Neutral", "Mobility", "Vision", "Hearing", "Mental Health"]

    fem, masc, ambig = [], [], []
    for kw in cond_keywords:
        match = flux_df[flux_df["Condition"].str.contains(kw, na=False, case=False)]
        if not match.empty:
            row = match.iloc[0]
            fem.append(int(row["Feminine presenting"]))
            masc.append(int(row["Masculine presenting"]))
            ambig.append(int(row["Ambiguous / indeterminate presentation"]))
        else:
            fem.append(0); masc.append(0); ambig.append(0)

    x     = np.arange(len(short_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, fem,   width, label="Feminine presenting",        color="#d62728")
    ax.bar(x,         masc,  width, label="Masculine presenting",       color="#1f77b4")
    ax.bar(x + width, ambig, width, label="Ambiguous / indeterminate",  color="#7f7f7f")

    ax.set_title("FLUX.1 Dev — Gender Drift Across Disability Conditions",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Image Count")
    ax.legend()

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig3_flux_gender_drift.png", dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {OUTPUT_DIR / 'fig3_flux_gender_drift.png'}")


if __name__ == "__main__":
    print("[*] Launching chart generation pipeline...")
    plot_gemini_refusal()
    plot_flux_gender_drift()
    print("[*] Pipeline complete")