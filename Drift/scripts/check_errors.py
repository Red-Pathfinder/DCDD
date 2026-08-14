import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your exact failed batch ID
batch_id = "batch_69e7a3e77e488190b0d0e2d207b7da34"

batch = client.batches.retrieve(batch_id)

print(f"Status: {batch.status}")

if batch.error_file_id:
    print("\n--- PULLING AUTOPSY REPORT (First 1000 chars) ---")
    error_content = client.files.content(batch.error_file_id).text
    print(error_content[:1000])
else:
    print("\nNo error file found. Check the dashboard manually.")