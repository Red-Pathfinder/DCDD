import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- OUTPUT CONFIGURATION ---
OUTPUT_IMAGE_PATH = r"C:\Study\Coding\Research\Drift\scripts\gender_inversion\flux_gender_inversion.png"

def generate_gender_inversion_chart():

    # Structural counts from FLUX contingency table matrix
    categories = ['Feminine Presenting', 'Masculine Presenting']
    baseline_counts = [95, 260]
    vision_counts = [190, 185]

    x = np.arange(len(categories))
    width = 0.35

    # Clean academic style
    sns.set_theme(style="whitegrid")

    # Slightly taller figure to accommodate bottom caption
    fig, ax = plt.subplots(figsize=(7, 5.8))

    # ---------------------------------------------------
    # Pastel, publication-friendly, high-contrast palette
    # ---------------------------------------------------
    color_baseline = '#A8C5E6'   # Soft periwinkle blue
    color_vision = '#F7C6A3'     # Soft apricot

    # Render grouped bars
    rects1 = ax.bar(
        x - width/2,
        baseline_counts,
        width,
        label='None (Neutral Baseline)',
        color=color_baseline,
        edgecolor='#444444',
        linewidth=0.8
    )

    rects2 = ax.bar(
        x + width/2,
        vision_counts,
        width,
        label='Vision Impairment Condition',
        color=color_vision,
        edgecolor='#444444',
        linewidth=0.8
    )

    # -----------------------
    # Main Title (Top Only)
    # -----------------------
    ax.set_title(
        'FLUX.1 Dev: Intersectional Gender Distribution Inversion',
        fontsize=13,
        fontweight='bold',
        pad=15
    )

    # Axis formatting
    ax.set_ylabel(
        'Generated Image Count',
        fontsize=11,
        fontweight='bold',
        labelpad=10
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        categories,
        fontsize=10.5,
        fontweight='semibold'
    )

    # Add headroom for labels
    ax.set_ylim(0, 310)

    # Grid refinement
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', visible=False)

    # -----------------------
    # Value Labels
    # -----------------------
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{int(height)}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#333333'
            )

    autolabel(rects1)
    autolabel(rects2)

    # -----------------------
    # Legend (Below Chart)
    # -----------------------
    ax.legend(
        title='Experimental Context Condition',
        title_fontsize=10,
        fontsize=10,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
        fancybox=True
    )

    # -----------------------
    # Bottom Caption
    # -----------------------
    fig.text(
        0.5,
        0.02,
        'Demonstrating Demographic Drift under Vision Impairment Modifiers',
        ha='center',
        fontsize=10,
        style='italic',
        color='#555555'
    )

    # Leave room for bottom caption
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save figure
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(
        OUTPUT_IMAGE_PATH,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"[+] Visualization generated at: {OUTPUT_IMAGE_PATH}")

    plt.show()


if __name__ == "__main__":
    generate_gender_inversion_chart()