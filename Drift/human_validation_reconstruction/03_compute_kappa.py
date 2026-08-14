import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

BASE_DIR = Path(r"C:\Study\Coding\Research\DCDD\Drift")
WORKING_DIR = BASE_DIR / "human_validation_reconstruction"
MANIFEST_PATH = WORKING_DIR / "sample_manifest.csv"
ANNOTATIONS_PATH = WORKING_DIR / "human_annotations.csv"
OUTPUT_CSV = WORKING_DIR / "table2_recomputed.csv"
OUTPUT_TEX = WORKING_DIR / "table2_recomputed.tex"
OUTPUT_MD = WORKING_DIR / "validation_statistics.md"

ATTRIBUTES = [
    "gender_presentation", "skin_tone", "sentiment",
    "physical_framing", "activity_engagement", "social_integration"
]

def main():
    if not MANIFEST_PATH.exists() or not ANNOTATIONS_PATH.exists():
        print("[!] ERROR: Manifest or Annotations CSV missing. Cannot compute kappa.")
        return

    manifest_df = pd.read_csv(MANIFEST_PATH)
    try:
        ann_df = pd.read_csv(ANNOTATIONS_PATH)
    except pd.errors.EmptyDataError:
        print("[!] Annotations file is empty. Please complete annotation first.")
        return

    if len(ann_df) == 0:
        print("[!] No annotations found.")
        return

    # We need to merge human annotations with VLM labels from manifest
    # manifest has columns: image_id, model, image_path, profession, age, disability_condition, [attributes]
    # ann_df has columns: image_id, model, attribute, human_label, timestamp

    results = []

    # Process overall and per-model
    models_to_evaluate = ['All Models'] + list(manifest_df['model'].unique())

    for model_name in models_to_evaluate:
        if model_name == 'All Models':
            model_manifest = manifest_df
            model_anns = ann_df
        else:
            model_manifest = manifest_df[manifest_df['model'] == model_name]
            model_anns = ann_df[ann_df['model'] == model_name]

        if len(model_anns) == 0:
            continue

        model_kappas = []
        for attr in ATTRIBUTES:
            # Get VLM labels for this model
            vlm_data = model_manifest[['image_id', attr]].rename(columns={attr: 'vlm_label'})
            
            # Get Human labels for this attribute and model
            attr_anns = model_anns[model_anns['attribute'] == attr][['image_id', 'human_label']]
            
            # Merge
            merged = pd.merge(vlm_data, attr_anns, on='image_id', how='inner')
            merged = merged.dropna(subset=['vlm_label', 'human_label'])

            if len(merged) == 0:
                continue

            vlm_labels = merged['vlm_label'].astype(str)
            human_labels = merged['human_label'].astype(str)

            # Compute stats
            n = len(merged)
            agreements = (vlm_labels == human_labels).sum()
            disagreements = n - agreements
            raw_agree_pct = (agreements / n) * 100

            try:
                kappa = cohen_kappa_score(vlm_labels, human_labels)
            except Exception:
                kappa = np.nan

            model_kappas.append(kappa)

            results.append({
                'Model': model_name,
                'Attribute': attr.replace('_', ' ').title(),
                'N': n,
                'Agreements': agreements,
                'Disagreements': disagreements,
                'Agreement %': raw_agree_pct,
                'Cohen\'s Kappa': kappa
            })

        # Macro average for the model
        if model_kappas:
            valid_kappas = [k for k in model_kappas if not np.isnan(k)]
            macro_kappa = np.mean(valid_kappas) if valid_kappas else np.nan
            results.append({
                'Model': model_name,
                'Attribute': 'Macro Average',
                'N': '',
                'Agreements': '',
                'Disagreements': '',
                'Agreement %': '',
                'Cohen\'s Kappa': macro_kappa
            })

    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV.name}")

    # Generate LaTeX Table (matches paper style)
    with open(OUTPUT_TEX, 'w') as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Attribute} & \\textbf{N} & \\textbf{Agreement} & \\textbf{Disagreements} & \\textbf{Cohen's $\\kappa$} \\\\\n")
        f.write("\\hline\n")
        
        current_model = ""
        for _, row in results_df.iterrows():
            model = row['Model']
            if model != current_model:
                if current_model != "":
                    f.write("\\hline\n")
                f.write(f"\\multirow{{{len(ATTRIBUTES) + 1}}}{{+}}{{ {model} }} ")
                current_model = model
            else:
                f.write(" & ")

            attr = row['Attribute']
            n = row['N']
            agree_pct = f"{row['Agreement %']:.1f}\\%" if row['Agreement %'] != '' else ''
            dis = row['Disagreements']
            kappa = f"{row['Cohen\'s Kappa']:.3f}" if pd.notna(row['Cohen\'s Kappa']) else "N/A"
            
            if attr == 'Macro Average':
                f.write(f"& \\textit{{{attr}}} & & & & \\textbf{{{kappa}}} \\\\\n")
            else:
                f.write(f"& {attr} & {n} & {agree_pct} & {dis} & {kappa} \\\\\n")
                
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Recomputed inter-rater reliability (Cohen's $\\kappa$) across blind human validation annotations.}\n")
        f.write("\\label{tab:cohens_kappa_recomputed}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {OUTPUT_TEX.name}")

    # Generate Markdown Summary
    with open(OUTPUT_MD, 'w') as f:
        f.write("# Validation Statistics\n\n")
        f.write("This report contains the recomputed Cohen's $\\kappa$ statistics from the blind annotation sample.\n\n")
        f.write("## Results\n\n")
        
        # We can format it nicely
        f.write(results_df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(f"Saved: {OUTPUT_MD.name}")


if __name__ == "__main__":
    main()
