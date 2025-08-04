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
- [Ollama](https://ollama.com/) (for running LLaMA 3 locally)
- Supabase (REST API)
- pdfplumber (PDF reading)

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SaadAnsari123/ScreenX.git
cd ScreenX-supabase

### **2. Install Python Dependencies**
'''bash
pip install -r requirements.txt

**### 3. Install and Run Ollama**
Download and install Ollama:
👉 https://ollama.com/download

Then in your terminal:
ollama run llama3

This will pull and run the llama3 model locally.
Note: The LLaMA 3 8B model is ~4.7 GB.

**🛠️ Configuration**
In the Python script (extracter_llama.py), replace the following:
SUPABASE_URL = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_API_KEY = "YOUR_SUPABASE_API_KEY"
SUPABASE_TABLE = "Resume_Details"
Ensure Supabase table fields match the extracted JSON structure:

candidate_name (text)
email (text)
phone (text)
skills (array or text[])
experience (integer)
highest_edu (text)
role_applied (text)
screened_on (timestamp)

Also, make sure Row Level Security (RLS) is disabled or allow INSERT for anonymous/public role.

**📄 Usage**
Put your PDF file in the repo folder and rename it if needed:

'''bash
python extracter_llama.py

It will:
Parse the resume
Ask LLaMA 3 to extract JSON
Upload data to Supabase
