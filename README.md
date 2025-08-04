ScreenX
ScreenX is an AI-powered resume parser and screening tool built using Python. It utilizes the LLaMA 3 large language model (running locally via Ollama) to extract structured candidate data from resume PDFs and automatically uploads the extracted information to a Supabase database.

Features
Resume Parsing:

Upload a resume in PDF format.

Extract structured candidate data, including:

candidate_name

email

phone

skills (as a list)

experience (years)

highest_edu

role_applied

screened_on (timestamp)

AI-Powered Extraction:

Uses the locally hosted LLaMA 3 model via Ollama for smart data extraction instead of traditional regex or keyword methods.

Supabase Integration:

Automatically uploads parsed data to a Supabase table.

Fully compatible with PostgreSQL and RESTful API.

PDF Resume Support:

Leverages the pdfplumber library to extract text cleanly from PDF documents.

Simple CLI Workflow:

Just run the script and everything works from parsing to uploading — no need for a UI.

Technologies Used
Core:

Python 3

Ollama for running LLaMA 3 locally

Supabase for backend data storage

Libraries:

pdfplumber for reading PDF text

requests for Supabase API communication

Other Tools:

LLaMA 3 (8B) model (~4.7 GB) via Ollama

Installation
Prerequisites
Python 3.x

Pip

Ollama (for running LLaMA 3)

A Supabase account + project setup

Steps
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/SaadAnsari123/ScreenX.git
cd ScreenX
Install Python Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Install and Run Ollama:

Download Ollama: 👉 https://ollama.com/download

Then run the following in your terminal:

bash
Copy
Edit
ollama run llama3
Configure Your Supabase Details:

In the extracter_llama.py file, update the following:

python
Copy
Edit
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_API_KEY = "YOUR_SUPABASE_API_KEY"
SUPABASE_TABLE = "Resume_Details"
Make sure your Supabase table includes:

candidate_name (text)

email (text)

phone (text)

skills (text[] or comma-separated text)

experience (integer)

highest_edu (text)

role_applied (text)

screened_on (timestamp)

✅ Ensure Row Level Security (RLS) is disabled or appropriate insert permissions are given to the public/anon role.

Run the Parser:

Place your resume PDF (e.g., resume.pdf) in the project folder and run:

bash
Copy
Edit
python extracter_llama.py
