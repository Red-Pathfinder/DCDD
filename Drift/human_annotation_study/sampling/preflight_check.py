"""
Pre-flight validation for the DCDD Human Annotation Study.
Checks all requirements before packaging annotator ZIPs.
"""
import re
import os

BASE = "/Users/red-pathfinder/Material Girl/Coding/DCDD/Drift/human_annotation_study"
HTML_PATH = os.path.join(BASE, "annotation_app", "index.html")

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  — {detail}")
        failed += 1

with open(HTML_PATH, "r") as f:
    html = f.read()

# ── TEST 1: Exactly 6 attributes ──
attr_count = html.count('id: "skin_tone"') + html.count('id: "gender_presentation"') + \
             html.count('id: "sentiment"') + html.count('id: "physical_framing"') + \
             html.count('id: "activity_engagement"') + html.count('id: "social_integration"')
check("TEST 1 — Exactly 6 attributes defined", attr_count == 6, f"found {attr_count}")

# Also verify no extra attribute ids
all_ids = re.findall(r'id:\s*"(\w+)"', html)
# Filter to only those in the attributes array (between 'const attributes' and '];')
attr_block = html[html.index("const attributes"):html.index("];", html.index("const attributes"))]
attr_ids = re.findall(r'id:\s*"(\w+)"', attr_block)
expected_ids = {"skin_tone", "gender_presentation", "sentiment", "physical_framing", "activity_engagement", "social_integration"}
check("TEST 1b — Attribute IDs match expected set", set(attr_ids) == expected_ids, f"found {set(attr_ids)}")

# ── TEST 2: Label validation — forbidden labels ──
forbidden = ["\"Medium\"", "\"Male\"", "\"Female\"", "\"Non-binary\"", "\"Safety Refusal\"",
             "\"Light\"", "\"Dark\"", "\"Idle/Observing\"", "\"Active/Working\"",
             "\"Group-Central\"", "\"Group-Peripheral\"", "\"Solo\"",
             "\"Expansive/Commanding\"", "\"Constricted/Withdrawn\""]
for label in forbidden:
    # Check only in the options arrays
    check(f"TEST 2 — {label} NOT in options", label not in attr_block, f"found in attribute definitions")

# ── TEST 2b: Required labels present ──
required_labels = [
    "Lighter skin tone", "Darker skin tone", "Ambiguous / indeterminate",
    "Masculine presenting", "Feminine presenting", "Ambiguous / indeterminate presentation",
    "Positive", "Negative", "Neutral", "Unclear",
    "Expansive/upright posture", "Constricted/hunched posture", "Unclear / indeterminate",
    "Active engagement", "Passive/dependent engagement",
    "Socially integrated", "Socially isolated"
]
for label in required_labels:
    check(f"TEST 2b — '{label}' present", label in attr_block, "missing from attribute definitions")

# ── TEST 3: Safety Refusal ──
# Count occurrences of 'Safety Refusal' in the entire options block
sr_count = attr_block.count("Safety Refusal")
check("TEST 3 — Safety Refusal selectable options = 0", sr_count == 0, f"found {sr_count}")

# ── TEST 4: Annotator IDs ──
check("TEST 4a — Annotator_1 option exists", '"Annotator_1"' in html)
check("TEST 4b — Annotator_2 option exists", '"Annotator_2"' in html)
check("TEST 4c — Annotator_3 option exists", '"Annotator_3"' in html)
check("TEST 4d — annotator_id in CSV header", '"annotator_id"' in html)

# ── TEST 5: Image set — all 300 ──
js_path = os.path.join(BASE, "annotation_app", "image_list.js")
with open(js_path, "r") as f:
    js = f.read()
img_ids_in_js = re.findall(r"'(img_[a-f0-9]+\.\w+)'", js)
check("TEST 5a — image_list.js has exactly 300 images", len(img_ids_in_js) == 300, f"found {len(img_ids_in_js)}")
check("TEST 5b — all unique", len(img_ids_in_js) == len(set(img_ids_in_js)))

# Verify blinded images directory
blind_dir = os.path.join(BASE, "blinded_images")
actual_files = set(os.listdir(blind_dir))
check("TEST 5c — blinded_images/ has 300 files", len(actual_files) == 300, f"found {len(actual_files)}")
check("TEST 5d — JS list matches blinded_images/ contents", set(img_ids_in_js) == actual_files,
      f"mismatch: {set(img_ids_in_js).symmetric_difference(actual_files)}")

# ── TEST 6: Blinding — no metadata in filenames ──
leak_words = ["flux", "sd35", "nanobanana", "neutral", "mobility", "vision", "hearing",
              "mental", "teacher", "doctor", "nurse", "professor", "adult", "senior", "young"]
leaks_found = []
for fn in actual_files:
    fn_lower = fn.lower()
    for w in leak_words:
        if w in fn_lower:
            leaks_found.append(f"{fn} contains '{w}'")
check("TEST 6 — No metadata leakage in filenames", len(leaks_found) == 0, "; ".join(leaks_found[:5]))

# ── TEST 7: Storage isolation ──
check("TEST 7 — localStorage keyed by annotator", "dcdd_annotations_${currentAnnotator}" in html or
      "dcdd_annotations_" in html)

# ── TEST 8: CSV schema ──
check("TEST 8a — CSV header uses 'image_id' (not 'anonymized_image_id')", '"image_id"' in html)
check("TEST 8b — CSV header uses 'sentiment' (not 'affective_sentiment')", '"sentiment"' in html and
      '"affective_sentiment"' not in html)
check("TEST 8c — 'data.sentiment' used in export", "data.sentiment" in html)

# ── TEST 9: No VLM/model metadata shown ──
# Check for any display of model names, VLM predictions
ui_forbidden = ["gpt-4", "vlm", "fal-ai", "flux/dev"]
for term in ui_forbidden:
    check(f"TEST 9 — '{term}' not in HTML", term.lower() not in html.lower())

print("\n" + "=" * 50)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 50)
if failed > 0:
    print("  *** DO NOT PACKAGE — FAILURES DETECTED ***")
else:
    print("  *** ALL CHECKS PASSED — SAFE TO PACKAGE ***")
