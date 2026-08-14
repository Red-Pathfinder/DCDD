import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ── Configuration ───────────────────────────────────────────────────────────
DATA_PATH = "attributes.csv"
OUT_DIR = "socioeconomic_drift_results"
os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# Aesthetic Colors for Clusters
CLUSTER_COLORS = {
    'High-Income / Specialized': '#2980b9',   # Blue
    'Mid-Tier / Professional': '#27ae60',     # Green
    'Service / Frontline': '#f39c12',        # Orange
    'Manual / Entry / Other': '#c0392b'       # Red
}
NEUTRAL_COLOR = '#95a5a6' # Consistent Gray for Neutral Baseline

# ── Professional Clusters Mapping ───────────────────────────────────────────
CLUSTER_MAP = {
    'High-Income / Specialized': [
        'Data Scientist', 'Doctor', 'Investment Analyst', 'Software Engineer', 'University Professor'
    ],
    'Mid-Tier / Professional': [
        'Accountant', 'Civil Servant', 'Graphic Designer', 'Journalist', 'Nurse', 'Physical Therapist', 'School Teacher'
    ],
    'Service / Frontline': [
        'Bank Teller', 'Chef', 'Hotel Receptionist', 'Police Officer', 'Firefighter', 'Waiter'
  ],
    'Manual / Entry / Other': [
        'Construction Worker', 'Electrician', 'Factory Worker', 'Farmer', 'Fisher', 'Retired Person', 'Unemployed Person'
    ]
}

rev_map = {prof: cluster for cluster, profs in CLUSTER_MAP.items() for prof in profs}

def plot_cluster_bars(df, attributes, cluster_name, results_df):
    """Generates a multi-panel figure of bar charts for a specific cluster."""
    sns.set_style("whitegrid")
    num_attrs = len(attributes)
    cols = 3
    rows = int(np.ceil(num_attrs / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle(f"Socioeconomic Bias Profile: {cluster_name}", fontsize=24, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    cluster_color = CLUSTER_COLORS.get(cluster_name, '#34495e')

    for i, attr in enumerate(attributes):
        ax = axes[i]
        sub_df = df[df['socioeconomic_cluster'] == cluster_name]
        
        # Calculate percentages for comparison
        counts = sub_df.groupby(['image_type', attr]).size().unstack(fill_value=0)
        
        # Ensure Neutral and Conditioned rows exist to avoid plotting errors
        for label in ['neutral', 'conditioned']:
            if label not in counts.index:
                counts.loc[label] = 0
        
        props = counts.div(counts.sum(axis=1), axis=0) * 100
        # Rigid reindexing to keep Neutral as the first bar in every pair
        props = props.reindex(['neutral', 'conditioned'])
        
        # Create grouped bar plot
        props.T.plot(kind='bar', ax=ax, color=[NEUTRAL_COLOR, cluster_color], width=0.8, edgecolor='white', linewidth=1)
        
        # Look up stats for the annotation
        row = results_df[(results_df['Cluster'] == cluster_name) & (results_df['Attribute'] == attr)].iloc[0]
        sig_star = "*" if row['FDR_P'] < 0.05 else ""
        ax.set_title(f"{attr.replace('_', ' ').title()}{sig_star}\nV={row['Cramers_V']:.2f} | CFR={row['CFR_%']:.1f}%", 
                     fontsize=14, fontweight='semibold', pad=10)
        
        ax.set_ylabel("Frequency (%)", fontsize=12)
        ax.set_xlabel("")
        ax.legend(title="", labels=['Neutral Baseline', 'Conditioned'], fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        sns.despine(ax=ax)

    # Remove empty plots from the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    file_safe_name = cluster_name.replace(' ', '_').replace('/', '')
    plt.savefig(os.path.join(OUT_DIR, f"bar_chart_{file_safe_name}.png"), bbox_inches='tight', dpi=150)
    plt.close()

# ── Main Analysis Execution ──────────────────────────────────────────────────
# 1. Load Data
df = pd.read_csv(DATA_PATH)
df['image_type'] = df['image_type'].str.strip().str.lower()
df['socioeconomic_cluster'] = df['profession'].map(rev_map)
df['pair_key'] = df['seed'].astype(str) + "_" + df['profession'] + "_" + df['age_bin']

# 2. Separate Meta-Data from Visual Attributes
metadata = ['dcdd_id', 'pair_id', 'seed', 'model', 'sector', 'profession', 'age_bin', 
            'image_type', 'disability', 'prompt', 'filename', 'timestamp', 
            'socioeconomic_cluster', 'pair_key']
attributes = [c for c in df.columns if c not in metadata]

# 3. Statistical Calculations
all_results = []
clusters = list(CLUSTER_MAP.keys())

for cluster in clusters:
    sub_df = df[df['socioeconomic_cluster'] == cluster]
    if sub_df.empty: continue

    for attr in attributes:
        # Cramér's V Calculation
        cross = pd.crosstab(sub_df['image_type'], sub_df[attr])
        v, p = 0.0, 1.0
        if not cross.empty and cross.shape[1] >= 2:
            chi2, p, _, _ = chi2_contingency(cross)
            n = cross.sum().sum()
            v = np.sqrt(chi2 / (n * min(cross.shape[0]-1, cross.shape[1]-1))) if n > 0 else 0.0

        # Flip Rate Calculation
        neutral_vals = sub_df[sub_df['image_type'] == 'neutral'].set_index('pair_key')[attr]
        cond_sub = sub_df[sub_df['image_type'] == 'conditioned']
        flips, valid = 0, 0
        for pk, c_val in zip(cond_sub['pair_key'], cond_sub[attr]):
            if pk in neutral_vals.index:
                n_v = str(neutral_vals.loc[pk]).lower().strip()
                c_v = str(c_val).lower().strip()
                if n_v not in ['nan', 'ambiguous'] and c_v not in ['nan', 'ambiguous']:
                    if n_v != c_v: flips += 1
                    valid += 1
        
        cfr = (flips / valid * 100) if valid > 0 else 0.0
        all_results.append({
            "Cluster": cluster, "Attribute": attr, 
            "Cramers_V": v, "Raw_P": p, "CFR_%": cfr
        })

results_df = pd.DataFrame(all_results)
# Apply Benjamini-Hochberg Correction across all tests
results_df["FDR_P"] = multipletests(results_df["Raw_P"], alpha=0.05, method="fdr_bh")[1]
results_df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

# 4. Generate Graphical Reports
for cluster in clusters:
    plot_cluster_bars(df, attributes, cluster, results_df)

print(f"Independent analysis complete. All visual reports saved in: {OUT_DIR}")