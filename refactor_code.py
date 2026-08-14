import os
import re
from pathlib import Path

# Paths to process
root_dir = Path(r"C:\Study\Coding\Research\DCDD\Supplementary Material")

def get_header(filename, subdir):
    # Mapping for headers
    headers = {
        "01_generate_flux.py": {
            "Purpose": "Generates FLUX.1 Dev images for the DCDD benchmark.",
            "Paper Sections": "Section 3.2",
            "Input Files": "None",
            "Output Files": "Generated images in 05_data/flux_pipeline/, 05_data/flux_metadata.csv",
            "Dependencies": "fal_client, dotenv, requests",
            "Approximate Runtime": "1-2 hours depending on API rate limits",
        },
        "02_generate_sd35.py": {
            "Purpose": "Generates Stable Diffusion 3.5 Large images for the DCDD benchmark.",
            "Paper Sections": "Section 3.2",
            "Input Files": "None",
            "Output Files": "Generated images in 05_data/sd35_pipeline/, 05_data/sd_metadata.csv",
            "Dependencies": "fal_client, dotenv, requests",
            "Approximate Runtime": "1-2 hours depending on API rate limits",
        },
        "03_nanabanana.py": {
            "Purpose": "Generates Nano Banana 2 (Gemini 3.1 Flash) images for the DCDD benchmark.",
            "Paper Sections": "Section 3.2",
            "Input Files": "None",
            "Output Files": "Generated images in 05_data/nanobanana_2_pipeline/, 05_data/gemini_batch_metadata_tracking.csv",
            "Dependencies": "google.generativeai, dotenv, requests",
            "Approximate Runtime": "1-2 hours",
        },
        "04_rescue_batch.py": {
            "Purpose": "Utility to rescue failed API batch generation runs.",
            "Paper Sections": "None (Optional Utility)",
            "Input Files": "Various",
            "Output Files": "Various",
            "Dependencies": "None",
            "Approximate Runtime": "N/A",
        },
        "01_flux_v2_attribute.py": {
            "Purpose": "Extracts visual attributes from FLUX generated images using GPT-5 mini.",
            "Paper Sections": "Section 3.3",
            "Input Files": "05_data/flux_metadata.csv, Generated images",
            "Output Files": "05_data/attributes_v2_flux.csv",
            "Dependencies": "openai, pydantic, dotenv",
            "Approximate Runtime": "10-15 minutes (OpenAI Batch API)",
        },
        "02_sd35_v2_attribute.py": {
            "Purpose": "Extracts visual attributes from SD 3.5 generated images using GPT-5 mini.",
            "Paper Sections": "Section 3.3",
            "Input Files": "05_data/sd_metadata.csv, Generated images",
            "Output Files": "05_data/attributes_v2_sd35.csv",
            "Dependencies": "openai, pydantic, dotenv",
            "Approximate Runtime": "10-15 minutes (OpenAI Batch API)",
        },
        "03_nanobanana_attribute.py": {
            "Purpose": "Extracts visual attributes from Nano Banana 2 generated images using GPT-4o-mini.",
            "Paper Sections": "Section 3.3",
            "Input Files": "05_data/gemini_batch_metadata_tracking.csv, Generated images",
            "Output Files": "05_data/nanobanana_attributes_final.csv",
            "Dependencies": "openai, pydantic, dotenv, PIL",
            "Approximate Runtime": "10-15 minutes (OpenAI Batch API)",
        },
        "09_profession_erasure_batch.py": {
            "Purpose": "Performs binary validation of profession adherence for FLUX and SD 3.5 images.",
            "Paper Sections": "Section 4.3",
            "Input Files": "05_data/flux_metadata.csv, 05_data/sd_metadata.csv",
            "Output Files": "results/profession_erasure_results.csv",
            "Dependencies": "openai, dotenv, PIL",
            "Approximate Runtime": "10-15 minutes (OpenAI Batch API)",
        },
        "human_audit.py": {
            "Purpose": "Interactive manual review queue to correct VLM ambiguous labels for Nano Banana 2.",
            "Paper Sections": "Section 3.4",
            "Input Files": "05_data/nanobanana_attributes_final.csv, 05_data/human_overrides.csv (optional)",
            "Output Files": "05_data/human_overrides.csv",
            "Dependencies": "pandas, PIL",
            "Approximate Runtime": "Interactive (depends on user)",
        },
        "human_validation_of_nanobanana_ambiguous_img.py": {
            "Purpose": "Read-only GUI viewer for auditing Nano Banana ambiguous images.",
            "Paper Sections": "Section 3.4 (Optional Tool)",
            "Input Files": "05_data/nanobanana_attributes_final.csv",
            "Output Files": "None",
            "Dependencies": "pandas, PIL",
            "Approximate Runtime": "Interactive",
        },
        "human_validation_of_sd_flux_non_amb.py": {
            "Purpose": "Interactive manual review queue to correct VLM ambiguous labels for FLUX and SD 3.5.",
            "Paper Sections": "Section 3.4",
            "Input Files": "05_data/attributes_v2_flux.csv, 05_data/attributes_v2_sd35.csv",
            "Output Files": "05_data/human_overrides.csv",
            "Dependencies": "pandas",
            "Approximate Runtime": "Interactive (depends on user)",
        },
        "vlm_alignment_flux.py": {
            "Purpose": "Empirical alignment audit sampling 50 confident predictions to benchmark human-VLM agreement.",
            "Paper Sections": "Section 3.4 (Validation)",
            "Input Files": "05_data/attributes_v2_flux.csv",
            "Output Files": "05_data/vlm_alignment_audit.csv, 05_data/human_overrides.csv",
            "Dependencies": "pandas",
            "Approximate Runtime": "Interactive (depends on user)",
        },
        "05_flux_analysis.py": {
            "Purpose": "Computes statistical significance, effect sizes, and CFR for the FLUX dataset.",
            "Paper Sections": "Section 4.1, 4.2",
            "Input Files": "05_data/attributes_v2_flux.csv",
            "Output Files": "05_data/statistical_summaries/flux_statistical_summary.csv, results/charts/flux charts/",
            "Dependencies": "pandas, numpy, scipy, statsmodels, matplotlib, seaborn",
            "Approximate Runtime": "< 1 minute",
        },
        "06_sd35_analysis.py": {
            "Purpose": "Computes statistical significance, effect sizes, and CFR for the SD 3.5 dataset.",
            "Paper Sections": "Section 4.1, 4.2",
            "Input Files": "05_data/attributes_v2_sd35.csv",
            "Output Files": "05_data/statistical_summaries/sd35_statistical_summary.csv, results/charts/sd 35 charts/",
            "Dependencies": "pandas, numpy, scipy, statsmodels, matplotlib, seaborn",
            "Approximate Runtime": "< 1 minute",
        },
        "03_nanobanana_analysis.py": {
            "Purpose": "Tri-model comparison combining FLUX, SD 3.5, and Nano Banana datasets.",
            "Paper Sections": "Section 5",
            "Input Files": "05_data/attributes_v2_flux.csv, 05_data/attributes_v2_sd35.csv, 05_data/nanobanana_attributes_final.csv",
            "Output Files": "results/DCDD_Final_Analysis_with_bias_severity.xlsx",
            "Dependencies": "pandas, numpy, scipy, statsmodels, openpyxl",
            "Approximate Runtime": "< 1 minute",
        },
        "03_nanobanana_charts.py": {
            "Purpose": "Generates algorithmic erasure and safety refusal charts.",
            "Paper Sections": "Section 5",
            "Input Files": "05_data/nanobanana_attributes_final.csv",
            "Output Files": "results/charts/nanobanana charts/fig1_gemini_safety_refusals.png",
            "Dependencies": "pandas, matplotlib, seaborn",
            "Approximate Runtime": "< 1 minute",
        },
        "13_explicit_analysis_no_ambiguous.py": {
            "Purpose": "Conducts statistical analysis exclusively on records with no ambiguous labels.",
            "Paper Sections": "Section 4.1 (Robustness Check)",
            "Input Files": "05_data/attributes_v2_flux.csv, 05_data/attributes_v2_sd35.csv, 05_data/nanobanana_attributes_final.csv",
            "Output Files": "results/DCDD_Explicit_Analysis_No_Ambiguous.xlsx",
            "Dependencies": "pandas, numpy, scipy, statsmodels, openpyxl",
            "Approximate Runtime": "< 1 minute",
        },
        "10_erasure_analysis.py": {
            "Purpose": "Analyzes and visualizes the Profession Erasure Rate across models.",
            "Paper Sections": "Section 4.3",
            "Input Files": "results/profession_erasure_results.csv",
            "Output Files": "results/DCDD_Executive_Summary_Profession_Erasure.xlsx",
            "Dependencies": "pandas, numpy, openpyxl",
            "Approximate Runtime": "< 1 minute",
        },
        "11_intersectional_analysis.py": {
            "Purpose": "Analyzes intersectional bias by fusing race and gender attributes.",
            "Paper Sections": "Section 4.4",
            "Input Files": "05_data/attributes_v2_flux.csv, 05_data/attributes_v2_sd35.csv",
            "Output Files": "results/DCDD_Intersectional_Bias_Results.xlsx",
            "Dependencies": "pandas, numpy, scipy, openpyxl",
            "Approximate Runtime": "< 1 minute",
        },
        "calc_flip_rate.py": {
            "Purpose": "Computes Counterfactual Flip Rate (CFR) matrices for all models.",
            "Paper Sections": "Section 4.2",
            "Input Files": "05_data/attributes_v2_flux.csv, 05_data/attributes_v2_sd35.csv, 05_data/nanobanana_attributes_final.csv",
            "Output Files": "05_data/flip_rates/*.csv",
            "Dependencies": "pandas, numpy",
            "Approximate Runtime": "< 1 minute",
        },
        "plot_flux_gender_inversion.py": {
            "Purpose": "Plots gender presentation drift figure using contingency counts.",
            "Paper Sections": "Section 4",
            "Input Files": "None (Hardcoded)",
            "Output Files": "results/charts/flux_gender_inversion.png",
            "Dependencies": "matplotlib",
            "Approximate Runtime": "< 1 minute",
        },
        "mental_health_collapse.py": {
            "Purpose": "Plots agentic attribute collapse for mental health conditions.",
            "Paper Sections": "Section 4",
            "Input Files": "None (Hardcoded)",
            "Output Files": "results/charts/mental_health_collapse.png",
            "Dependencies": "matplotlib",
            "Approximate Runtime": "< 1 minute",
        },
    }
    
    data = headers.get(filename)
    if not data:
        data = {
            "Purpose": f"Execution script for {filename}",
            "Paper Sections": "N/A",
            "Input Files": "Various",
            "Output Files": "Various",
            "Dependencies": "pandas, numpy",
            "Approximate Runtime": "Varies"
        }
        
    return f'''"""
Purpose:
{data["Purpose"]}

Paper Sections:
{data["Paper Sections"]}

Input Files:
{data["Input Files"]}

Output Files:
{data["Output Files"]}

Dependencies:
{data["Dependencies"]}

Approximate Runtime:
{data["Approximate Runtime"]}
"""

'''

def replace_paths_and_add_header(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Prepend pathlib import if needed
    if "from pathlib import Path" not in content and "import pathlib" not in content:
        # Find first import
        import_match = re.search(r"^(import |from )", content, flags=re.MULTILINE)
        if import_match:
            idx = import_match.start()
            content = content[:idx] + "from pathlib import Path\n" + content[idx:]
        else:
            content = "from pathlib import Path\n\n" + content
            
    # Add BASE_DIR if not present
    if "BASE_DIR =" not in content:
        # insert after imports
        import_block_end = 0
        for m in re.finditer(r"^(import |from ).*$", content, flags=re.MULTILINE):
            import_block_end = max(import_block_end, m.end())
        content = content[:import_block_end] + "\n\n# Dynamic relative path to supplementary root\nBASE_DIR = Path(__file__).resolve().parent.parent\n" + content[import_block_end:]

    # Fix hardcoded absolute paths with regex replacements
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\01_flux_attributes_v2\\attributes_v2\.csv["\'](\))?', r'BASE_DIR / "05_data" / "attributes_v2_flux.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\02_sd35_attributes_v2\\attributes_v2\.csv["\'](\))?', r'BASE_DIR / "05_data" / "attributes_v2_sd35.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\03_nanobanana_attributes\\nanobanana_attributes_final\.csv["\'](\))?', r'BASE_DIR / "05_data" / "nanobanana_attributes_final.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\03_nanobanana_attributes\\nanobanana_attributes_v2_validated\.csv["\'](\))?', r'BASE_DIR / "05_data" / "nanobanana_attributes_final.csv"', content)
    
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\flux_pipeline\\metadata\.csv["\'](\))?', r'BASE_DIR / "05_data" / "flux_metadata.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\sd35_pipeline\\metadata\.csv["\'](\))?', r'BASE_DIR / "05_data" / "sd_metadata.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\nanobanana_2_pipeline\\batch_metadata_tracking\.csv["\'](\))?', r'BASE_DIR / "05_data" / "gemini_batch_metadata_tracking.csv"', content)
    
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\flux_pipeline["\'](\))?', r'BASE_DIR / "05_data" / "flux_pipeline"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\sd35_pipeline["\'](\))?', r'BASE_DIR / "05_data" / "sd35_pipeline"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_raw\\nanobanana_2_pipeline["\'](\))?', r'BASE_DIR / "05_data" / "nanobanana_2_pipeline"', content)
    
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\human_overrides\.csv["\'](\))?', r'BASE_DIR / "05_data" / "human_overrides.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\vlm_alignment_audit\.csv["\'](\))?', r'BASE_DIR / "05_data" / "vlm_alignment_audit.csv"', content)
    
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\01_flux_attributes_v2["\'](\))?', r'BASE_DIR / "05_data"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\02_sd35_attributes_v2["\'](\))?', r'BASE_DIR / "05_data"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\data_processed\\02_nanobanana_attributes["\'](\))?', r'BASE_DIR / "05_data"', content)
    
    # Generic replacements for results folder
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\results\\(.*?)\.xlsx["\'](\))?', r'BASE_DIR / "results" / "\2.xlsx"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\results\\(.*?)\.csv["\'](\))?', r'BASE_DIR / "results" / "\2.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\results\\charts\\(.*?)\.csv["\'](\))?', r'BASE_DIR / "results" / "charts" / "\2.csv"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\results\\charts\\(.*?)["\'](\))?', r'BASE_DIR / "results" / "charts" / "\2"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\results["\'](\))?', r'BASE_DIR / "results"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\results["\'](\))?', r'BASE_DIR / "results"', content)
    content = re.sub(r'(Path\()?r?["\']C:\\Study\\Coding\\Research\\Drift\\scripts\\flip rate["\'](\))?', r'BASE_DIR / "05_data" / "flip_rates"', content)
    
    content = re.sub(r'BASE_DIR = Path\(__file__\)\.resolve\(\)\.parents\[3\]', r'BASE_DIR = Path(__file__).resolve().parent.parent', content)
    content = re.sub(r'BASE_DIR = os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)', r'BASE_DIR = Path(__file__).resolve().parent.parent', content)
    
    # Fix os.path.join formatting
    content = re.sub(r'os\.path\.join\(BASE_DIR,\s*"([^"]+)",\s*"([^"]+)"\)', r'BASE_DIR / "\1" / "\2"', content)
    content = re.sub(r'os\.path\.join\(BASE_DIR,\s*"([^"]+)"\)', r'BASE_DIR / "\1"', content)
    content = re.sub(r'os\.path\.join\(OUTPUT_DIR,\s*"([^"]+)"\)', r'OUTPUT_DIR / "\1"', content)

    # Specific fix
    content = re.sub(r'BASE_DIR / "data_raw" / "sd35_pipeline"', r'BASE_DIR / "05_data" / "sd35_pipeline"', content)
    content = re.sub(r'BASE_DIR / "data_raw" / "flux_pipeline"', r'BASE_DIR / "05_data" / "flux_pipeline"', content)
    content = re.sub(r'BASE_DIR / "data_raw" / "nanobanana_2_pipeline"', r'BASE_DIR / "05_data" / "nanobanana_2_pipeline"', content)
    
    # Prevent duplicate BASE_DIR declarations
    base_dir_lines = [line for line in content.split('\n') if line.startswith('BASE_DIR = ')]
    if len(base_dir_lines) > 1:
        # Keep the first one, remove the rest
        first_found = False
        new_lines = []
        for line in content.split('\n'):
            if line.startswith('BASE_DIR = '):
                if not first_found:
                    new_lines.append(line)
                    first_found = True
                # else skip
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)

    # Finally prepend the header
    header = get_header(file_path.name, file_path.parent.name)
    final_content = header + content
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_content)

for py_file in root_dir.rglob("*.py"):
    replace_paths_and_add_header(py_file)
    print(f"Processed {py_file.name}")
