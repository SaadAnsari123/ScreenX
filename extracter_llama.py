import re
import pdfplumber
import requests
import json
from datetime import datetime

# --- CONFIG ---
SUPABASE_URL = "https://rawijfjlyspjmjqxoitj.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhd2lqZmpseXNwam1qcXhvaXRqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3MDkxNjEsImV4cCI6MjA2OTI4NTE2MX0.U7HfqWZLcp0MBTokQQp5qn9cxoodDtfdgG6u4xiwroc"
SUPABASE_TABLE = "Resume_Details"
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama local API

HEADERS = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# --- LLaMA 3 Prompt ---
LLAMA_PROMPT_TEMPLATE = """
Extract the following fields from the resume text below and return JSON:
- candidate_name
- email
- phone
- skills (as an array)
- experience (in years, integer)
- highest_edu (highest degree)
- role_applied
Use null for any missing field.

Resume Text:
===
{resume_text}
===
"""

# --- Extract Resume Text ---
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# --- Extract JSON Block from LLaMA Output ---
def extract_json_from_llama_response(response_text):
    match = re.search(r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        raise ValueError("No valid JSON block found in LLaMA response.")

# --- Ask LLaMA 3 for Structured Data ---
def extract_resume_data_with_llama3(text):
    prompt = LLAMA_PROMPT_TEMPLATE.format(resume_text=text)
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        content = response.json()["response"]
        try:
            return extract_json_from_llama_response(content)
        except Exception as e:
            print("❌ Failed to extract JSON from model output:", e)
            print("📄 Model Output:", content)
            return None
    else:
        print("❌ LLaMA API error:", response.text)
        return None

# --- Upload to Supabase ---
def upload_to_supabase(data):
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    response = requests.post(url, headers=HEADERS, data=json.dumps(data))
    print("🔁 Uploading to:", url)
    print("📦 Data:", json.dumps(data, indent=2))
    if response.status_code == 201:
        print("✅ Uploaded successfully.")
    else:
        print("❌ Upload failed:")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

# --- MAIN ---
if __name__ == "__main__":
    resume_path = "sample_resume.pdf"
    text = extract_text_from_pdf(resume_path)
    extracted_data = extract_resume_data_with_llama3(text)
    if extracted_data:
        extracted_data["screened_on"] = datetime.utcnow().isoformat()
        upload_to_supabase(extracted_data)
