import os
import json
import base64
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Study\Coding\Research\Drift\data_raw\nanobanana_2_pipeline"
JOB_ID = "batches/bnzgouihym6qobnivcd45ef0oxugokbulg64"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def rescue_images():
    print("="*60)
    print(f"RESCUING BATCH JOB: {JOB_ID}")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Fetch the specific job
    print(f"Connecting to Google servers to check status...")
    try:
        job = client.batches.get(name=JOB_ID)
        state = job.state.name
        print(f"Current Job State: {state}")
        
        # --- NEW PROGRESS TRACKER ---
        if hasattr(job, 'request_counts') and job.request_counts:
            total = getattr(job.request_counts, 'total', 'Unknown')
            completed = getattr(job.request_counts, 'completed', 0)
            failed = getattr(job.request_counts, 'failed', 0)
            print(f"Progress: {completed} / {total} images processed ({failed} failures)")
        else:
            print("Progress: Tracking metrics not yet available from Google.")
        # ----------------------------

    except Exception as e:
        print(f"[ERROR] Could not connect to Google API: {e}")
        return

    # 2. Handle the State
    if state == 'JOB_STATE_SUCCEEDED':
        print("\nJob is finished! Downloading the massive results package...")
        results = client.files.download(file=job.dest.file_name).decode('utf-8')
        
        extracted = 0
        print("\nExtracting PNGs... (Printing every 100th image to save terminal lag)")
        for line in results.splitlines():
            if not line: continue
            res = json.loads(line)
            
            if 'response' in res and res['response']:
                try:
                    img_b64 = res['response']['candidates'][0]['content']['parts'][0]['inlineData']['data']
                    filepath = os.path.join(OUTPUT_DIR, f"{res['key']}.png")
                    
                    with open(filepath, 'wb') as f:
                        f.write(base64.b64decode(img_b64))
                    
                    extracted += 1
                    if extracted % 100 == 0:
                        print(f"  ✓ Extracted and saved {extracted} images so far...")
                        
                except (KeyError, IndexError):
                    print(f"  ✗ Failed to parse image data for {res.get('key', 'unknown')}")
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE. Successfully extracted {extracted} images to your folder.")
        print("="*60)
        
    elif state in ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING', 'JOB_STATE_IN_PROGRESS']:
        print("\nJob is still processing on Google's end.")
        print("Action: Wait a few minutes and run this script again to check progress.")
        
    else:
        print(f"\n[WARNING] Job ended with state: {state}. Cannot extract images.")
        if hasattr(job, 'error'):
            print(f"Error details: {job.error}")

if __name__ == "__main__":
    rescue_images()