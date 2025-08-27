import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import simpledialog
from pathlib import Path
import asyncio
import aiohttp
import orjson
import pypdfium2 as pdfium
import threading
import re
import requests
import numpy as np
import datetime

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # Change to any model you have
OLLAMA_TIMEOUT = 120

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"   # run once: ollama pull nomic-embed-text

OUTPUT_DIR = Path("processed_resumes")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_TEXT_FILE = OUTPUT_DIR / "resumes_raw.arrow"  # Stores all resume text

# Context storage
resumes_text = ""  # Combined text from all resumes
candidate_names = []  # Just for reference


# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a single PDF using pypdfium2"""
    try:
        doc = pdfium.PdfDocument(pdf_path)
        text_parts = []
        for page in doc:
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            text_parts.append(text)
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")


# --- OLLAMA COMMUNICATION ---
async def query_ollama(prompt: str, callback):
    """Send prompt to Ollama and stream response"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        }
        try:
            async with asyncio.timeout(OLLAMA_TIMEOUT):
                async with session.post(OLLAMA_URL, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        callback(f"❌ Ollama Error {resp.status}: {error_text}")
                        return

                    async for line in resp.content:
                        if line:
                            try:
                                chunk = orjson.loads(line)
                                if "response" in chunk:
                                    token = chunk["response"]
                                    callback(token)
                            except Exception:
                                pass
        except asyncio.TimeoutError:
            callback("❌ Request timed out. Try a shorter question.")
        except Exception as e:
            callback(f"❌ Connection failed: {str(e)}")


# --- GUI APPLICATION ---
class ResumeChatbotApp:
    KNOWN_SKILLS = {
        "python","java","javascript","typescript","c","c++","go","rust",
        "django","flask","fastapi","spring","react","node","express",
        "sql","postgres","mysql","mongodb","pandas","numpy","scikit-learn","pytorch","tensorflow",
        "git","docker","kubernetes","terraform","linux","bash","ci","cd",
        "aws","azure","gcp","rest","graphql"
    }

    def __init__(self, root):
        self.root = root
        self.root.title("💬 Resume Chatbot")
        self.root.geometry("900x700")
        self.is_processing = False

        self.resume_map = {}  # name -> raw resume text

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        ttk.Label(main_frame, text="💬 Resume Chatbot", style="Header.TLabel").pack(pady=10)
        ttk.Label(main_frame, text="Select a folder with resumes, then ask questions about candidates.").pack()

        # --- Control Frame ---
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=10)

        self.load_btn = ttk.Button(ctrl_frame, text="📁 Load Resumes Folder", command=self.load_resumes)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # ⭐ Rank button
        self.rank_btn = ttk.Button(ctrl_frame, text="⭐ Rank Resumes", command=self.rank_resumes)
        self.rank_btn.pack(side=tk.LEFT, padx=5)
        self.rank_btn.config(state="disabled")  # Enable only after resumes load

        self.status_label = ttk.Label(ctrl_frame, text="Ready", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # --- Chat Area ---
        chat_frame = ttk.LabelFrame(main_frame, text="Chat with Resumes", padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state="disabled", height=20,
                                                  font=("Arial", 10))
        self.chat_area.pack(fill=tk.BOTH, expand=True)

        # --- Input Area ---
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = tk.Entry(input_frame, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.send_question())

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_question)
        self.send_btn.pack(side=tk.RIGHT)

        # Enable input only after loading resumes
        self.user_input.config(state="disabled")
        self.send_btn.config(state="disabled")

    def load_resumes(self):
        if self.is_processing:
            return

        folder = filedialog.askdirectory(title="Select Folder with Resume PDFs")
        if not folder:
            return

        folder_path = Path(folder)
        pdf_files = [f for f in folder_path.glob("*.pdf") if f.is_file()]
        if not pdf_files:
            messagebox.showwarning("No Files", "No PDF files found in selected folder.")
            return

        self.is_processing = True
        self.load_btn.config(state="disabled")
        self.status_label.config(text="Processing resumes...")

        # Run in background thread
        threading.Thread(target=self._process_resumes, args=(pdf_files,), daemon=True).start()

    def _process_resumes(self, pdf_files):
        global resumes_text, candidate_names
        full_text = []
        candidate_names = []
        self.resume_map = {}

        for i, pdf in enumerate(pdf_files):
            try:
                text = extract_text_from_pdf(str(pdf))
                name = text.splitlines()[0].strip() if text else pdf.stem
                candidate_names.append(name)
                self.resume_map[name] = text
                full_text.append(f"--- Resume: {name} ({pdf.name}) ---\n{text}")
            except Exception as e:
                full_text.append(f"--- Resume: {pdf.name} ---\n[Failed to extract: {e}]")

            self.root.after(0, lambda i=i: self.status_label.config(
                text=f"Processed {i + 1}/{len(pdf_files)} resumes..."))

        resumes_text = "\n\n".join(full_text)
        self.root.after(0, self.on_resumes_loaded)

    def on_resumes_loaded(self):
        self.is_processing = False
        self.load_btn.config(state="normal")
        self.user_input.config(state="normal")
        self.send_btn.config(state="normal")
        self.rank_btn.config(state="normal")
        self.status_label.config(text=f"✅ Loaded {len(self.resume_map)} resumes. You can now ask questions.")
        self.append_message("🤖 Bot", "Hello! I've loaded the resumes. You can now ask questions like:\n"
                                      "- 'Who knows Python?'\n"
                                      "- 'List candidates with AWS experience'\n"
                                      "- 'Show me contact info for Alex'\n"
                                      "Or click ⭐ Rank Resumes to compare candidates with a job description.")

    def send_question(self):
        question = self.user_input.get().strip()
        if not question:
            return

        self.append_message("You", question)
        self.user_input.delete(0, tk.END)
        self.send_btn.config(state="disabled")
        threading.Thread(target=self._query_async, args=(question,), daemon=True).start()

    def _query_async(self, question):
        prompt = f"""
You are an AI assistant that answers questions based ONLY on the following resume data.

<context>
{resumes_text}
</context>

Question: {question}

Instructions:
- Answer clearly and concisely.
- If information is not available, say "Not enough information".
- Do not invent details.
- Keep answers under 5 sentences.
"""

        def token_callback(token):
            self.root.after(0, lambda: self.append_token(token))

        asyncio.run(query_ollama(prompt, token_callback))
        self.root.after(0, lambda: self.send_btn.config(state="normal"))

    def append_message(self, sender: str, msg: str):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"\n{sender}: {msg}\n", "user" if sender == "You" else "bot")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def append_token(self, token: str):
        self.chat_area.config(state="normal")
        if self.chat_area.get("end-2c", "end") == "\n\n":
            self.chat_area.insert(tk.END, "🤖 Bot: ")
        self.chat_area.insert(tk.END, token)
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    # ---------------------------
    # ⭐ Resume Ranking Feature
    # ---------------------------
    def rank_resumes(self):
        global candidate_names

        if not self.resume_map:
            messagebox.showwarning("No Data", "Please load resumes first.")
            return

        job_desc = simpledialog.askstring("Job Description", "Enter the job description:")
        if not job_desc:
            return

        try:
            jd_vec = self._embed(job_desc)
            ranked = []
            for name in candidate_names:
                resume_text = self.resume_map.get(name, "")
                if not resume_text.strip():
                    ranked.append((0, name))
                    continue

                res_vec = self._embed(resume_text)
                sim = self._cosine(jd_vec, res_vec)
                kw = self._skill_match_score(job_desc, resume_text)
                final = 0.85 * sim + 0.15 * kw
                score_10 = int(round(10 * final))
                ranked.append((score_10, name))

            ranked.sort(reverse=True, key=lambda x: x[0])
            self.append_message("🤖 Bot", "📋 Ranked Resumes (semantic + keywords):")
            for score, name in ranked:
                self.append_message("🤖 Bot", f"{name}: {score}/10")

        except Exception:
            # fallback LLM scoring
            self.append_message("🤖 Bot", "Embedding model not available; falling back to LLM scoring.")
            ranked = []
            for name in candidate_names:
                resume_text = self.resume_map.get(name, "")
                score = self._rank_with_ollama(job_desc, resume_text)
                ranked.append((score, name))
            ranked.sort(reverse=True, key=lambda x: x[0])
            for score, name in ranked:
                self.append_message("🤖 Bot", f"{name}: {score}/10")

    def _rank_with_ollama(self, job_desc, resume_text):
        prompt = f"""
You are an expert recruiter. Rate this resume from 1 (worst match) to 10 (perfect match) for the given job description.

Job Description:
{job_desc}

Resume:
{resume_text}

Only respond with the score (1-10).
"""
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            result = resp.json()
            return int(str(result.get("response", "")).strip())
        except Exception as e:
            print("LLM ranking error:", e)
            return 0

    # ---------------------------
    # Embedding + keyword helpers
    # ---------------------------
    def _embed(self, text: str) -> np.ndarray:
        resp = requests.post(OLLAMA_EMBED_URL, json={
            "model": EMBED_MODEL,
            "prompt": text
        }, timeout=60)
        resp.raise_for_status()
        vec = resp.json().get("embedding")
        if not vec:
            raise RuntimeError("No embedding returned")
        return np.array(vec, dtype=float)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _extract_skills(self, text: str) -> set:
        tokens = re.findall(r"[A-Za-z\+\#\.]+", text.lower())
        cleaned = [t.strip(".") for t in tokens]
        return {t for t in cleaned if t in self.KNOWN_SKILLS}

    def _skill_match_score(self, jd: str, res: str) -> float:
        jd_s = self._extract_skills(jd)
        rs_s = self._extract_skills(res)
        if not jd_s:
            return 0.0
        overlap = len(jd_s & rs_s)
        return min(overlap, 10) / 10.0


# --- START APPLICATION ---
if __name__ == "__main__":
    print(" Make sure Ollama is running: open terminal and run `ollama serve`")
    print(f" Make sure you have pulled the models: `ollama pull {OLLAMA_MODEL}`, `ollama pull {EMBED_MODEL}`")

    root = tk.Tk()
    app = ResumeChatbotApp(root)
    root.mainloop()
