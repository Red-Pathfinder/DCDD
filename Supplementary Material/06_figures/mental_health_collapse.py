"""
Purpose:
Plots agentic attribute collapse for mental health conditions.

Paper Sections:
Section 4

Input Files:
None (Hardcoded)

Output Files:
results/charts/mental_health_collapse.png

Dependencies:
matplotlib

Approximate Runtime:
< 1 minute
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Dynamic relative path to supplementary root
BASE_DIR = Path(__file__).resolve().parent.parent


# --- CONFIGURATION & PATHS ---
GEMINI_DATA_PATH = BASE_DIR / "05_data" / "nanobanana_attributes_final.csv"
OUTPUT_IMAGE_PATH = BASE_DIR / "06_figures" / "mental_health_collapse.png"

ATTRIBUTES_OF_INTEREST = [
    "sentiment", "gaze_direction", "physical_framing", "activity_engagement", "social_integration"
]

DISPLAY_LABELS = [
    'Affective Sentiment', 
    'Gaze Direction', 
    'Physical Framing\n(Posture)', # Added a newline breakout to keep text from overlapping horizontally
    'Activity Engagement', 
    'Social Integration'
]

def generate_collapse_chart():
    print(f"[*] Loading validation tracking from: {GEMINI_DATA_PATH}")
    if not GEMINI_DATA_PATH.exists():
        print(f"[!] Target data file not found at: {GEMINI_DATA_PATH}")
        return

    df = pd.read_csv(GEMINI_DATA_PATH)
    
    type_col = 'image_type' if 'image_type' in df.columns else 'condition'
    disability_col = 'disability' if 'disability' in df.columns else 'condition'
    
    df_neutral = df[df[type_col].astype(str).str.lower() == 'neutral'].copy()
    df_conditioned = df[df[type_col].astype(str).str.lower() != 'neutral'].copy()
    
    baseline_map = {}
    for _, row in df_neutral.iterrows():
        pid = row['pair_id']
        baseline_map[pid] = {attr: str(row[attr]).strip().lower() for attr in ATTRIBUTES_OF_INTEREST if attr in df.columns}
        
    def clean_condition(val):
        val = str(val).lower().strip()
        if 'mobility' in val: return 'Mobility Impairment'
        if 'vision' in val: return 'Vision Impairment'
        if 'hearing' in val: return 'Hearing Impairment'
        if 'mental' in val or 'psych' in val: return 'Mental Health Condition'
        return 'Other'

    df_conditioned['clean_disability'] = df_conditioned[disability_col].apply(clean_condition)
    conditions = ['Mobility Impairment', 'Vision Impairment', 'Hearing Impairment', 'Mental Health Condition']
    
    results = {cond: {attr: [] for attr in ATTRIBUTES_OF_INTEREST} for cond in conditions}
    for _, row in df_conditioned.iterrows():
        pid = row['pair_id']
        cond = row['clean_disability']
        
        if cond not in conditions or pid not in baseline_map:
            continue
        if str(row[ATTRIBUTES_OF_INTEREST[0]]).strip().lower() == 'safety refusal':
            continue
            
        base_line = baseline_map[pid]
        for attr in ATTRIBUTES_OF_INTEREST:
            curr_val = str(row[attr]).strip().lower()
            base_val = base_line[attr]
            if 'safety refusal' in curr_val or 'safety refusal' in base_val:
                continue
                
            results[cond][attr].append(1 if curr_val != base_val else 0)
            
    plot_df = pd.DataFrame(index=DISPLAY_LABELS)
    for cond in conditions:
        rates = []
        for attr in ATTRIBUTES_OF_INTEREST:
            arr = results[cond][attr]
            rates.append((sum(arr) / len(arr) * 100) if len(arr) > 0 else 0.0)
        plot_df[cond] = rates

    # --- CHOSEN ACADEMIC TYPOGRAPHY & DESIGN ---
    sns.set_theme(style="ticks") # Switched to clean ticks layout for a sharp academic profile
    fig, ax = plt.subplots(figsize=(10.5, 6)) # Adjusted dimensions to avoid squeezed borders
    
    x = np.arange(len(plot_df.index))
    width = 0.18

    # Professional tailored color palette (Muted, cohesive controls paired with an explicit warning red)
    colors = {
        'Mobility Impairment': '#5A6B7C',      # Muted Slate Gray
        'Vision Impairment': '#7A9A82',        # Sage Green
        'Hearing Impairment': '#D1A153',       # Patina Ochre Gold
        'Mental Health Condition': '#C0392B'   # High-Contrast Crimson
    }

    rects1 = ax.bar(x - 1.5*width, plot_df['Mobility Impairment'], width, label='Mobility Impairment', color=colors['Mobility Impairment'], edgecolor='#34495E', linewidth=0.6)
    rects2 = ax.bar(x - 0.5*width, plot_df['Vision Impairment'], width, label='Vision Impairment', color=colors['Vision Impairment'], edgecolor='#4D5F50', linewidth=0.6)
    rects3 = ax.bar(x + 0.5*width, plot_df['Hearing Impairment'], width, label='Hearing Impairment', color=colors['Hearing Impairment'], edgecolor='#7E6535', linewidth=0.6)
    rects4 = ax.bar(x + 1.5*width, plot_df['Mental Health Condition'], width, label='Mental Health Condition', color=colors['Mental Health Condition'], edgecolor='#7B241C', linewidth=1.2)

    # Label styling and axis boundaries
    ax.set_ylabel('Counterfactual Flip Rate (%)', fontsize=11, fontweight='bold', labelpad=10)
    
    # Title is nested with ample spacing pad to isolate it from data bars
    ax.set_title('Gemini Flash 3.1 Contextual Dimension Drift Profile\nIsolating the "Mental Health Collapse" Signature Across Parallel Prompts', fontsize=12, fontweight='bold', pad=25)
    
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontsize=10, fontweight='semibold')
    
    # FIX 1: Raised limits to 120% to completely clear text obstructions at the top of the chart
    ax.set_ylim(0, 120) 
    
    # Clean structural grids (Only thin horizontal helpers, no messy vertical lines)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')
    ax.set_axisbelow(True)
    sns.despine(trim=True) # Strips away harsh top and right outer axis lines

    # FIX 2: Annotate text labels clearly above the bars with a colored background patch if needed
    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#7B241C')

    # FIX 3: Placed legend on upper-center horizon to clear bar heights across all categories
    ax.legend(title='Disability Constraint Modifier', title_fontsize='10', loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7E9')
    
    plt.tight_layout()
    
    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_IMAGE_PATH), dpi=300)
    print(f"[+] Cleaned publication-ready visualization exported to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    generate_collapse_chart()