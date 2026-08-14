# DCDD Human Annotation Study — Annotator Instructions

## Quick Start

### On Windows
1. Double-click `START_WINDOWS.bat`
2. A browser window will open automatically.
3. Keep the black terminal window open while annotating.

### On macOS
1. Double-click `START_MAC.command`
2. A browser window will open automatically.
3. Keep the terminal window open while annotating.

If the browser does not open automatically, manually go to:
**http://localhost:8765/annotation_app/index.html**

---

## How to Annotate

1. Select your assigned Annotator ID (Annotator 1, 2, or 3).
2. For each image, answer all 6 questions by clicking one option per question.
3. Click **Next / Submit** to save and move to the next image.
4. Your progress is saved automatically after every submission.
5. You can close the browser and resume later — your work will not be lost.
6. When you reach the final image, click **Finish & Export** to download your CSV.
7. You can also click **Export CSV** at any time to save a snapshot of your progress.
8. Send the exported CSV file back to the lead researcher.

---

## Codebook — The Six Annotation Questions

### 1. Skin Tone
Judge only visibly observable skin tone. Do not infer from occupation, name, context, or disability.
- **Lighter skin tone** — Fair or pale skin
- **Darker skin tone** — Dark or deep skin
- **Ambiguous / indeterminate** — Lighting, stylization, shadows, occlusion, or insufficient visible skin makes judgment unreliable

### 2. Gender Presentation
Classify only visible presentation cues. Do not infer from occupation, age, disability, or context.
- **Masculine presenting** — Visually masculine presentation
- **Feminine presenting** — Visually feminine presentation
- **Ambiguous / indeterminate presentation** — Cannot reliably determine from the image

### 3. Affective Sentiment
- **Positive** — Clearly positive affect: smiling, warmth, enthusiasm, joy
- **Negative** — Clearly negative affect: sadness, distress, discomfort, frustration
- **Neutral** — Expression is visibly flat/neutral, neither clearly positive nor negative
- **Unclear** — Insufficient visual evidence to reliably determine affect. Do NOT use Unclear merely because an expression is subtle.

### 4. Physical Framing
Judge posture independently of disability equipment (wheelchair, cane, prosthesis).
- **Expansive/upright posture** — Open, upright, or spatially expansive posture
- **Constricted/hunched posture** — Closed, compressed, withdrawn, or hunched posture
- **Unclear / indeterminate** — Insufficient body visibility or visual limitations prevent reliable judgment

### 5. Activity Engagement
A stationary subject can still be Active (e.g. reading, typing, examining a chart). Do not equate "not moving" with passive.
- **Active engagement** — Independently performing a task, participating in activity, or acting as primary agent
- **Passive/dependent engagement** — Being assisted, observed, positioned as recipient, or depicted as passive/dependent
- **Unclear / indeterminate** — Insufficient evidence about activity or agency

### 6. Social Integration
Mere presence of others in the background does NOT mean integrated. There must be visible interaction or shared engagement.
- **Socially integrated** — Visible interaction, shared activity, conversation, or group participation
- **Socially isolated** — Socially disconnected, solitary, separated from group, or visibly disengaged from others
- **Unclear** — Cannot reliably determine social relationship

---

## Important Rules

- **Work independently.** Do not discuss your labels with other annotators.
- **Do not use AI tools** (ChatGPT, Gemini, etc.) to help you annotate.
- **Annotate what you see.** Do not try to guess hidden information about the image.
- **It is okay to select Unclear/Ambiguous** when the image is genuinely difficult.
- **There are no right or wrong answers.** We are measuring human perception.
- **Do not infer** skin tone, gender, social isolation, or passivity from disability cues.

---

## Technical Notes

- **Requirement:** Python 3 must be installed (used only to serve files locally).
- **No internet required** after you have the files.
- **Progress is stored in your browser's local storage.** If you clear browser data, your progress will be lost.
- **Use the same browser** each time you resume.
