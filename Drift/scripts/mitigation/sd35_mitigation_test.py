"""
Universal Semantic Anchoring Mitigation Script for DCDD Paper
Model: Stable Diffusion 3.5 Large via fal-ai
Strategy: Universal Sequence-Locked Anchoring + Role-Based Negative CFG
"""

import os
import json
import time
import requests
import pandas as pd
import fal_client
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── PATHS ──────────────────────────────────────────────────────────────────────
CSV_PATH    = Path(r"C:\Study\Coding\Research\Drift\data_processed\02_sd35_attributes\attributes.csv")
OUTPUT_DIR  = Path(r"C:\Study\Coding\Research\Drift\data_raw\sd35_mitigation_test")
LOG_PATH    = OUTPUT_DIR / "mitigation_log_test.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_ATTRIBUTE_COLS = [
    "gender_presentation", "perceived_race", "skin_tone", "perceived_age",
    "sentiment", "environment_type", "professional_attire", "gaze_direction",
    "physical_framing", "activity_engagement", "social_integration"
]

# ─── UNIVERSAL MECHANISTIC ANCHORING ──────────────────────────────────────────
def build_universal_prompts(profession: str, age_bin: str, disability: str):
    prof_key = profession.lower().strip()
    
    # 1. POSITIVE: Rely on the model's semantic prior for the profession's location
    positive_prompt = (
        f"a candid photograph of a {age_bin} {prof_key} actively performing their job "
        f"in their typical work environment, wearing standard professional attire for their role, "
        f"competent, focused, natural active posture, clearly defined facial features, "
        f"{disability}, "
        f"photorealistic, highly detailed"
    )

    # 2. NEGATIVE: Target the "Patient/Helpless" bias, not the "Hospital" location
    negative_prompt = (
        "patient, medical gown, lying in bed, sick, idle, being assisted, "
        "passive, helpless, hunched, ambiguous race, blurred features, "
        "hyper-artificial, glossy stock photo, superhero"
    )

    return positive_prompt, negative_prompt

# ─── DRIFT SCORE & PAIR MATCHING ──────────────────────────────────────────────
def identify_top_biased_pairs(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    neutral_df     = df[df["image_type"] == "neutral"].copy()
    conditioned_df = df[df["image_type"] == "conditioned"].copy()
    records = []

    for keys, n_group in neutral_df.groupby(["seed", "profession", "age_bin"]):
        seed, profession, age_bin = keys
        c_group = conditioned_df[
            (conditioned_df["seed"] == seed) &
            (conditioned_df["profession"] == profession) &
            (conditioned_df["age_bin"] == age_bin)
        ]

        if c_group.empty: continue  
        n_row = n_group.iloc[0]
        
        for _, c_row in c_group.iterrows():
            score = 0
            for col in VISUAL_ATTRIBUTE_COLS:
                n_val = n_row.get(col)
                c_val = c_row.get(col)
                if pd.notna(n_val) and pd.notna(c_val):
                    if str(n_val).strip().lower() != str(c_val).strip().lower():
                        score += 1
                        
            records.append({
                "seed": seed,
                "profession": profession,
                "age_bin": age_bin,
                "disability": c_row.get("disability", "unknown"),
                "drift_score": score,
            })

    pairs_df = pd.DataFrame(records).sort_values("drift_score", ascending=False)
    return pairs_df.head(top_n).reset_index(drop=True)

# ─── GENERATION PIPELINE ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DCDD Universal Mitigation Pipeline - 10 IMAGE TEST RUN")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    
    top_pairs = identify_top_biased_pairs(df, top_n=10)
    print(f"[*] Selected Top 10 drifted pairs. Max Drift Score: {top_pairs['drift_score'].max()}")

    log_entries = []
    
    for idx, row in top_pairs.iterrows():
        profession = row["profession"]
        age_bin    = row["age_bin"]
        disability = row["disability"]
        seed       = int(row["seed"])
        drift      = row["drift_score"]

        prompt, neg_prompt = build_universal_prompts(profession, age_bin, disability)
        print(f"  [{idx+1:02d}/10] Generating: {profession} | {disability}")

        safe_prof = profession.replace(' ', '_').replace('/', '')
        safe_dis  = disability.replace(' ', '_').replace('/', '')
        filename = f"{idx+1:02d}_{safe_prof}_{age_bin}_{safe_dis}_seed{seed}_mitigated.jpeg"
        img_path = OUTPUT_DIR / filename

        try:
            def print_status(update):
                print("    [fal.ai] Processing...", end="\r")

            result = fal_client.subscribe(
                "fal-ai/stable-diffusion-v35-large",
                arguments={
                    "prompt": prompt,
                    "negative_prompt": neg_prompt,
                    "seed": seed,
                    "image_size": "square_hd",
                    "num_inference_steps": 28,
                    "guidance_scale": 7.5
                },
                on_queue_update=print_status
            )
            
            print("                                                  ", end="\r")
            
            image_url = result["images"][0]["url"]
            img_data = requests.get(image_url, timeout=30).content
            
            with open(img_path, "wb") as f:
                f.write(img_data)
                
            log_entries.append({
                "pair_key": f"{seed}_{profession}_{age_bin}",
                "profession": profession,
                "disability": disability,
                "seed": seed,
                "drift_score": int(drift),
                "mitigated_prompt": prompt,
                "negative_prompt": neg_prompt,
                "filename": filename,
                "status": "success"
            })
            
            print(f"    [✓] Saved: {filename}")
            
        except Exception as e:
            print(f"\n    [!] FAILED: {e}")

        time.sleep(1) 

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=4)

    print("\n" + "=" * 70)
    print(f"✨ TEST COMPLETE. Check: {OUTPUT_DIR.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()