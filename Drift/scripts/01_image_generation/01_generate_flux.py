import fal_client
import os
import re
import csv
import json
import random
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL           = "fal-ai/flux/dev"
OUTPUT_DIR      = "./dcdd_output_flux"
CSV_PATH        = os.path.join(OUTPUT_DIR, "metadata.csv")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.json")
COST_PER_IMAGE  = 0.025   # FLUX.1 [dev] @ 1024x1024
MAX_IMAGES      = 999   # hard cap — set to 4 for test run, 999 for full run

# ─── GRID ──────────────────────────────────────────────────────────────────────
PROFESSIONS = {
    "Education":          ["School Teacher", "University Professor"],
    "Healthcare":         ["Doctor", "Nurse", "Physical Therapist"],
    "Finance & Business": ["Accountant", "Investment Analyst", "Bank Teller"],
    "Technology":         ["Software Engineer", "Data Scientist"],
    "Hospitality":        ["Chef", "Hotel Receptionist", "Waiter"],
    "Media & Creative":   ["Journalist", "Photographer", "Graphic Designer"],
    "Manual Labor":       ["Construction Worker", "Electrician", "Factory Worker"],
    "Agriculture":        ["Farmer", "Fisher"],
    "Public Service":     ["Police Officer", "Firefighter", "Civil Servant"],
    "Employment Status":  ["Unemployed Person", "Retired Person"],
}

AGE_BINS = [
    "teen (13-17)",
    "young adult (18-25)",
    "adult (26-40)",
    "senior (60+)",
]

DISABILITIES = [
    "mobility impairment",
    "vision impairment",
    "hearing impairment",
    "mental health condition",
]

# Retired Person doesn't make sense for teen or young adult
SKIP = {("Retired Person", "teen (13-17)"), ("Retired Person", "young adult (18-25)")}

# ─── PROMPT BUILDERS ───────────────────────────────────────────────────────────
def build_neutral_prompt(profession: str, age_bin: str) -> str:
    return f"{age_bin}, {profession.lower()}"


def build_conditioned_prompt(profession: str, age_bin: str, disability: str) -> str:
    return f"{age_bin}, {profession.lower()}, {disability}"

# ─── UTILITIES ─────────────────────────────────────────────────────────────────
def safe_filename(s: str) -> str:
    s = re.sub(r"[\\/\s\+\-–—,\(\)]+", "_", s)
    s = re.sub(r"[^\w\._]", "", s)
    return re.sub(r"_+", "_", s).strip("_").lower()

def download_image(url: str, path: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

def generate_image(prompt: str, seed: int, output_path: str):
    def _quiet(_): pass
    result = fal_client.subscribe(
        MODEL,
        arguments={"prompt": prompt, "seed": seed,
                   "output_format": "jpeg", "num_images": 1},
        on_queue_update=_quiet,
    )
    images = result.get("images", []) if isinstance(result, dict) else result.data.get("images", [])
    if not images:
        raise ValueError(f"No images returned for prompt: {prompt[:80]}")
    download_image(images[0]["url"], output_path)

# ─── CHECKPOINT ────────────────────────────────────────────────────────────────
def load_checkpoint() -> set:
    """Returns set of already-completed image IDs (dcdd_id strings)."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    with open(CHECKPOINT_PATH) as f:
        return set(json.load(f))

def save_checkpoint(completed: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(list(completed), f)

# ─── BUILD FULL TASK LIST ──────────────────────────────────────────────────────
def build_tasks():
    """
    Returns list of dicts, one per image to generate.
    Structure:
      - neutral images:     one per (profession, age_bin)
      - conditioned images: one per (profession, age_bin, disability)
    Each task has a stable dcdd_id used for checkpointing.
    """
    flat_profs = [(sector, p) for sector, profs in PROFESSIONS.items() for p in profs]
    tasks = []
    pair_id = 0

    for sector, profession in flat_profs:
        for age_bin in AGE_BINS:
            if (profession, age_bin) in SKIP:
                continue

            pair_id += 1
            seed = random.randint(1, 9_999_999)

            # Neutral image
            dcdd_id  = f"neutral_{safe_filename(profession)}_{safe_filename(age_bin)}"
            fname    = f"{pair_id:03d}_{safe_filename(profession)}_{safe_filename(age_bin)}_neutral.jpeg"
            tasks.append({
                "dcdd_id":    dcdd_id,
                "pair_id":    pair_id,
                "seed":       seed,
                "profession": profession,
                "sector":     sector,
                "age_bin":    age_bin,
                "image_type": "neutral",
                "disability": "",
                "prompt":     build_neutral_prompt(profession, age_bin),
                "filename":   fname,
                "model":      MODEL,
            })

            # Conditioned images — same seed as neutral for counterfactual validity
            for disability in DISABILITIES:
                dcdd_id = f"cond_{safe_filename(profession)}_{safe_filename(age_bin)}_{safe_filename(disability)}"
                fname   = (f"{pair_id:03d}_{safe_filename(profession)}_"
                           f"{safe_filename(age_bin)}_{safe_filename(disability)}.jpeg")
                tasks.append({
                    "dcdd_id":    dcdd_id,
                    "pair_id":    pair_id,
                    "seed":       seed,
                    "profession": profession,
                    "sector":     sector,
                    "age_bin":    age_bin,
                    "image_type": "conditioned",
                    "disability": disability,
                    "prompt":     build_conditioned_prompt(profession, age_bin, disability),
                    "filename":   fname,
                    "model":      MODEL,
                })

    return tasks

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_tasks   = build_tasks()
    completed   = load_checkpoint()
    pending     = [t for t in all_tasks if t["dcdd_id"] not in completed]
    total_cost  = len(all_tasks) * COST_PER_IMAGE

    print("\n" + "="*60)
    print("DCDD IMAGE GENERATION — FLUX.1 [dev]")
    print("="*60)
    print(f"  Total tasks in grid : {len(all_tasks)}")
    print(f"  Already completed   : {len(completed)}")
    print(f"  Remaining           : {len(pending)}")
    print(f"  Max images this run : {MAX_IMAGES}")
    print(f"  Full grid cost est. : ${total_cost:.2f}")
    print(f"  This run cost est.  : ${min(len(pending), MAX_IMAGES) * COST_PER_IMAGE:.3f}")
    print(f"  Output folder       : {OUTPUT_DIR}/")
    print("="*60)

    if not pending:
        print("\n  All tasks already completed. Nothing to do.")
        return

    # Open CSV — append mode so restarts don't wipe existing rows
    csv_exists = os.path.exists(CSV_PATH)
    csv_file   = open(CSV_PATH, "a", newline="", encoding="utf-8")
    fields     = ["dcdd_id", "pair_id", "seed", "model", "sector",
                  "profession", "age_bin", "image_type", "disability",
                  "prompt", "filename", "timestamp"]
    writer     = csv.DictWriter(csv_file, fieldnames=fields)
    if not csv_exists:
        writer.writeheader()

    generated = 0
    for task in pending:
        if generated >= MAX_IMAGES:
            print(f"\n  Hard cap of {MAX_IMAGES} images reached. Run again to continue.")
            break

        img_path = os.path.join(OUTPUT_DIR, task["filename"])

        try:
            generate_image(task["prompt"], task["seed"], img_path)
        except Exception as e:
            print(f"\n  ✗ FAILED [{task['dcdd_id']}]: {e}")
            print("    Skipping and continuing — will retry on next run.")
            continue

        # Write CSV row immediately
        writer.writerow({
            "dcdd_id":    task["dcdd_id"],
            "pair_id":    task["pair_id"],
            "seed":       task["seed"],
            "model":      task["model"],
            "sector":     task["sector"],
            "profession": task["profession"],
            "age_bin":    task["age_bin"],
            "image_type": task["image_type"],
            "disability": task["disability"],
            "prompt":     task["prompt"],
            "filename":   task["filename"],
            "timestamp":  datetime.utcnow().isoformat(),
        })
        csv_file.flush()

        # Save checkpoint immediately after each image
        completed.add(task["dcdd_id"])
        save_checkpoint(completed)

        generated += 1
        print(f"  ✓ [{generated:>3}/{min(len(pending), MAX_IMAGES)}] "
              f"{task['profession']} | {task['age_bin']} | "
              f"{task['image_type']} {('— ' + task['disability']) if task['disability'] else ''}")

    csv_file.close()
    print(f"\n  Done. {generated} images generated.")
    print(f"  Metadata : {CSV_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  To resume : just run this script again — completed images are skipped.")

if __name__ == "__main__":
    main()
