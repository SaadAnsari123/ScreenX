# 🧠 AI Resume Parser & Screener with LLaMA 3 + Supabase

This project is a Python-based tool that extracts structured data from resume PDFs using the **LLaMA 3** language model (via [Ollama](https://ollama.com/)) and uploads the results to a **Supabase** database.

---

## 📌 Features

- ✅ Extracts:
  - `candidate_name`
  - `email`
  - `phone`
  - `skills` (as a list)
  - `experience` (years)
  - `highest_edu`
  - `role_applied`
- 🤖 Uses **LLaMA 3** locally for smarter extraction (compared to regex-based methods)
- 📄 Supports PDF resumes using `pdfplumber`
- ☁️ Automatically uploads parsed data to a Supabase table

---

## 🧰 Tech Stack

- Python 3
- [Ollama](https://ollama.com/) – for running LLaMA 3 locally
- Supabase (PostgreSQL + REST API)
- pdfplumber – for PDF text extraction

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SaadAnsari123/ScreenX.git
cd ScreenX

### 2. Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should include:

nginx
Copy
Edit
pdfplumber
requests
3. Install and Run Ollama
Download Ollama from:
👉 https://ollama.com/download

Then open your terminal and run:

bash
Copy
Edit
ollama run llama3
🧠 This will download (~4.7 GB) and run the LLaMA 3 model locally.

⚙️ Configuration
In the extracter_llama.py script, update the following:

python
Copy
Edit
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_API_KEY = "YOUR_SUPABASE_API_KEY"
SUPABASE_TABLE = "Resume_Details"
Make sure your Supabase table has the following fields:

candidate_name (text)

email (text)

phone (text)

skills (text[] or comma-separated text)

experience (integer)

highest_edu (text)

role_applied (text)

screened_on (timestamp)

✅ Also ensure:

Row Level Security (RLS) is disabled, or

A policy allows INSERT for the anon/public role.

📄 Usage
Place your PDF file (e.g., sample_resume.pdf) in the project folder, then run:

bash
Copy
Edit
python extracter_llama.py
This will:

Extract text from the resume

Use LLaMA 3 to convert it to structured JSON

Upload the result to your Supabase table
