import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import plotly.graph_objects as go

# ── Setup ───────────────────────────────────────────────────────────────────
DATA_PATH = "attributes.csv"
OUT_DIR = "disability_drift_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# Consistent Aesthetic Mapping
DISABILITY_COLORS = {
    'mobility impairment': '#e74c3c',
    'vision impairment': '#2ecc71',
    'hearing impairment': '#f1c40f',
    'mental health condition': '#9b59b6'
}
NEUTRAL_COLOR = '#3498db'

def load_and_prep():
    df = pd.read_csv(DATA_PATH)
    df['image_type'] = df['image_type'].str.strip().str.lower()
    metadata = ['dcdd_id', 'pair_id', 'seed', 'model', 'sector', 'profession', 
                'age_bin', 'image_type', 'disability', 'prompt', 'filename', 'timestamp']
    attributes = [c for c in df.columns if c not in metadata]
    df['pair_key'] = df['seed'].astype(str) + "_" + df['profession'] + "_" + df['age_bin']
    disabilities = df[df['image_type'] == 'conditioned']['disability'].unique().tolist()
    return df, attributes, disabilities

def calculate_metrics(df, attribute, disability_name):
    sub_df = df[(df['image_type'] == 'neutral') | 
                ((df['image_type'] == 'conditioned') & (df['disability'] == disability_name))]
    cross = pd.crosstab(sub_df['image_type'], sub_df[attribute])
    chi2, p, _, _ = chi2_contingency(cross) if not cross.empty else (0, 1, 0, 0)
    n = cross.sum().sum()
    v = np.sqrt(chi2 / (n * min(cross.shape[0]-1, cross.shape[1]-1))) if n > 0 else 0.0

    neutral_lookup = df[df['image_type'] == 'neutral'].set_index('pair_key')[attribute]
    cond_subset = df[(df['image_type'] == 'conditioned') & (df['disability'] == disability_name)]
    flips, valid = 0, 0
    for pk, c_val in zip(cond_subset['pair_key'], cond_subset[attribute]):
        if pk in neutral_lookup:
            n_val = str(neutral_lookup.loc[pk]).lower().strip()
            if n_val not in ['nan', 'ambiguous'] and str(c_val).lower().strip() not in ['nan', 'ambiguous']:
                if n_val != str(c_val).lower().strip(): flips += 1
                valid += 1
    cfr = (flips / valid * 100) if valid > 0 else 0.0
    return v, p, cfr

# ── Global Visualization Logic ──────────────────────────────────────────────
df, attributes, disabilities = load_and_prep()
all_results = []

for dis in disabilities:
    res = [dict(zip(["Attribute", "Cramers_V", "Raw_P", "CFR_%"], [a, *calculate_metrics(df, a, dis)])) for a in attributes]
    res_df = pd.DataFrame(res)
    res_df["FDR_P"] = multipletests(res_df["Raw_P"], alpha=0.05, method="fdr_bh")[1]
    res_df["Disability"] = dis
    all_results.append(res_df)

master_res = pd.concat(all_results)

# 1. Comparative Heatmap
plt.figure(figsize=(10, 8))
pivot_v = master_res.pivot(index="Attribute", columns="Disability", values="Cramers_V")
sns.heatmap(pivot_v, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title("Comparative Heatmap: Cramér's V (Bias Intensity)", fontsize=14)
plt.savefig(os.path.join(OUT_DIR, "comparative_heatmap.png"), bbox_inches='tight', dpi=150)

# 2. Dot Plot (CFR Distribution)
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
sns.stripplot(data=master_res, x="CFR_%", y="Attribute", hue="Disability", 
              palette=list(DISABILITY_COLORS.values()), size=10, alpha=0.8)
plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='20% Flip Threshold')
plt.title("Counterfactual Flip Rate (CFR) across Disabilities", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(OUT_DIR, "cfr_dot_plot.png"), bbox_inches='tight', dpi=150)

print(f"Updates complete. All visualisations saved to {OUT_DIR}.")