# ScreenX

ScreenX is an AI-powered resume parser and screening tool built using Python. It utilizes the LLaMA 3 large language model (running locally via Ollama) to extract structured candidate data from resume PDFs and automatically uploads the extracted information to a Supabase database.

---

## Features

1. **Resume Parsing:**

-Upload a resume in PDF format.
-Extract structured candidate data, including:
  candidate_name
  email
  phone
  skills (as a list)
  experience (years)
  highest_edu
  role_applied
  screened_on (timestamp)

2. **AI-Powered Extraction:**

-Uses the locally hosted LLaMA 3 model via Ollama for smart data extraction instead of traditional regex or keyword methods.

3. **Supabase Integration:**

-Automatically uploads parsed data to a Supabase table.
-Fully compatible with PostgreSQL and RESTful API.

4. **PDF Resume Support:**

-Leverages the pdfplumber library to extract text cleanly from PDF documents.

5. **Simple CLI Workflow:**

-Just run the script and everything works from parsing to uploading — no need for a UI.

6. **Technologies Used**

 **Core:**
  -Python 3
  -Ollama for running LLaMA 3 locally
  -Supabase for backend data storage

 **Libraries:**
  -pdfplumber for reading PDF text
  -requests for Supabase API communication
  
**Other Tools:**
  -LLaMA 3 (8B) model (~4.7 GB) via Ollama

**Installation**
_**Prerequisites**_

  -Python 3.x
  -Pip
  -Ollama (for running LLaMA 3)
  -A Supabase account + project setup

**Steps**
1.**Clone the Repository:**

bash
git clone https://github.com/SaadAnsari123/ScreenX.git
cd ScreenX

2.**Install Python Dependencies:**

bash
pip install -r requirements.txt

3.**Install and Run Ollama:**

Download Ollama: 👉 https://ollama.com/download

Then run the following in your terminal:

bash
ollama run llama3

4.**Configure Your Supabase Details:**

In the extracter_llama.py file, update the following:

SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_API_KEY = "YOUR_SUPABASE_API_KEY"
SUPABASE_TABLE = "Resume_Details"

_Make sure your Supabase table includes:_

  -candidate_name (text)
  -email (text)
  -phone (text)
  -skills (text[] or comma-separated text)
  -experience (integer)
  -highest_edu (text)
  -role_applied (text)
  -screened_on (timestamp)

*✅ Ensure Row Level Security (RLS) is disabled or appropriate insert permissions are given to the public/anon role.**

5.**Run the Parser:**

Place your resume PDF (e.g., resume.pdf) in the project folder and run:

bash
python extracter_llama.py
