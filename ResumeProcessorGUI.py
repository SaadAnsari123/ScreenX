import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from pathlib import Path
import orjson
import pypdfium2 as pdfium
import threading
import re
import requests
import numpy as np
import datetime
import subprocess
import sys
import os

# --- CONFIGURATION ---
OLLAMA_MODEL = "llama3.2"           # Must be pulled: ollama pull llama3.2
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Must be pulled: ollama pull nomic-embed-text
OLLAMA_HTTP_URL = "http://localhost:11434/api/generate"   # Default Ollama port (not 8080!)
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_TIMEOUT = 120

OUTPUT_DIR = Path("processed_resumes")
OUTPUT_DIR.mkdir(exist_ok=True)
RAW_TEXT_FILE = OUTPUT_DIR / "resumes_raw.arrow"

# Context storage
resumes_text = ""
candidate_names = []

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(pdf_path: str) -> str:
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

# --- OLLAMA COMMUNICATION: HTTP + CLI FALLBACK ---
def query_ollama_local_fallback(prompt: str, callback):
    """
    Try HTTP first; if fails, fall back to ollama CLI subprocess.
    Works offline if model is pulled locally.
    Uses 'ollama generate --model <model> --prompt ... --stream'
    """
    # Try HTTP first
    try:
        import requests
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True
        }
        resp = requests.post(OLLAMA_HTTP_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                try:
                    chunk = orjson.loads(line)
                    if "response" in chunk:
                        token = chunk["response"]
                        if token:
                            callback(token)
                except (orjson.JSONDecodeError, KeyError):
                    continue
        return

    except Exception as http_err:
        print(f"HTTP failed ({http_err}), falling back to ollama CLI...", file=sys.stderr)

    # Fallback: Use ollama CLI subprocess
    try:
        # Ensure ollama is in PATH
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            callback("❌ 'ollama' CLI not found. Install from https://ollama.com")
            return

        # Run ollama generate --stream
        cmd = [
            "ollama", "generate",
            "--model", OLLAMA_MODEL,
            "--prompt", prompt,
            "--stream"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output line by line
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = orjson.loads(line)
                if "response" in chunk:
                    token = chunk["response"]
                    if token:
                        callback(token)
            except (orjson.JSONDecodeError, KeyError):
                # Sometimes ollama outputs non-JSON lines (like progress), ignore
                continue

        # Wait for completion
        _, stderr = process.communicate()
        if process.returncode != 0:
            callback(f"❌ CLI Error: {stderr.strip()}")
        return

    except FileNotFoundError:
        callback("❌ 'ollama' command not found. Install Ollama and pull your model.")
    except Exception as e:
        callback(f"❌ CLI failed: {str(e)}")

# --- EMBEDDINGS: LOCAL HTTP ONLY (NO FALLBACK) ---
def _embed(text: str) -> np.ndarray:
    """
    Uses Ollama's embedding API. If this fails, we raise an error.
    For offline use: ensure 'nomic-embed-text' is pulled: ollama pull nomic-embed-text
    Alternative: replace with TF-IDF if no embedding model available (see comment below).
    """
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json={
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text
        }, timeout=60)
        resp.raise_for_status()
        vec = resp.json().get("embedding")
        if not vec:
            raise RuntimeError("No embedding returned")
        return np.array(vec, dtype=float)
    except Exception as e:
        raise RuntimeError(
            f"❌ Embedding failed: {str(e)}\n"
            f"Make sure you've pulled the model: ollama pull {OLLAMA_EMBED_MODEL}\n"
            f"Or switch to TF-IDF fallback (commented out in code)."
        )

# --- COSINE SIMILARITY & SKILL EXTRACTION ---
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

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
        self.resume_map = {}
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

        self.status_label = ttk.Label(ctrl_frame, text="Ready", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # --- Chat Area ---
        chat_container = ttk.LabelFrame(main_frame, text="Chat with Resumes", padding="10")
        chat_container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.chat_area = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            state="disabled",
            height=16,
            font=("Arial", 14)
        )
        self.chat_area.tag_config("user", foreground="black", font=("Arial", 14))
        self.chat_area.tag_config("bot", foreground="darkblue", font=("Arial", 14, "bold"))
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # --- Input Area ---
        input_frame = ttk.Frame(chat_container)
        input_frame.pack(fill=tk.X)

        self.user_input = tk.Entry(input_frame, font=("Arial", 12), width=50)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda e: self.send_question())

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_question)
        self.send_btn.pack(side=tk.RIGHT)

        # Disable until loaded
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
            messagebox.showwarning("No Files", "No PDF files found.")
            return

        self.is_processing = True
        self.load_btn.config(state="disabled")
        self.status_label.config(text="Processing resumes...")
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
                full_text.append(f"[Error reading {pdf.name}: {e}]")

            self.root.after(0, lambda i=i: self.status_label.config(
                text=f"Processed {i + 1}/{len(pdf_files)} resumes..."))

        resumes_text = "\n\n".join(full_text)
        self.root.after(0, self.on_resumes_loaded)

    def on_resumes_loaded(self):
        self.is_processing = False
        self.load_btn.config(state="normal")
        self.user_input.config(state="normal")
        self.send_btn.config(state="normal")
        
        self.status_label.config(text=f"✅ Loaded {len(self.resume_map)} resumes.")
        self.append_message("🤖 Bot", "Hello! Ask things like:\n\n"
                                      "• Who knows Python?\n"
                                      "• Show me AWS experience\n"
                                      "• Rank these candidates")

    def send_question(self):
        question = self.user_input.get().strip()
        if not question:
            return
        self.append_message("You", question)
        self.user_input.delete(0, tk.END)
        self.send_btn.config(state="disabled")
        # Synchronous thread call — no asyncio!
        threading.Thread(target=self._query_sync, args=(question,), daemon=True).start()

    def _query_sync(self, question):
        """Synchronous version of query — uses local Ollama via HTTP or CLI fallback."""
        prompt = f"""
You are an AI assistant that answers questions based ONLY on the following resume data.

<context>
{resumes_text}
</context>

Question: {question}

Instructions:
- Answer clearly and concisely.
- If info missing, say "Not enough information".
- Keep under 5 sentences.
"""

        def token_callback(token):
            self.root.after(0, lambda: self.append_token(token))

        # This function now handles both HTTP and CLI fallback synchronously
        query_ollama_local_fallback(prompt, token_callback)

        self.root.after(0, lambda: self.send_btn.config(state="normal"))

    def append_message(self, sender: str, msg: str):
        self.chat_area.config(state="normal")
        self.chat_area.insert(tk.END, f"\n{sender}: ", "user" if sender == "You" else "bot")
        self.chat_area.insert(tk.END, msg + "\n")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    def append_token(self, token: str):
        self.chat_area.config(state="normal")
        if self.chat_area.get("end-2c", "end") == "\n\n":
            self.chat_area.insert(tk.END, "🤖 Bot: ", "bot")
        self.chat_area.insert(tk.END, token)
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)

    # ---------------------------
    # ⭐ Resume Ranking Feature
    # ---------------------------

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


if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeChatbotApp(root)
    root.mainloop()
