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
from PIL import Image, ImageTk
import time
import json

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
    # Try HTTP first with optimized settings
    try:
        import requests
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": 2048,  # Reduced context window for faster processing
                "temperature": 0.7,  # Slightly higher temperature for faster responses
                "top_p": 0.9,  # Limit token selection for faster generation
                "num_predict": 256  # Limit response length for faster completion
            }
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

    # Fallback: Use ollama CLI subprocess with optimized parameters
    try:
        # Ensure ollama is in PATH
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            callback("❌ 'ollama' CLI not found. Install from https://ollama.com")
            return

        # Run ollama generate --stream with optimized parameters
        cmd = [
            "ollama", "generate",
            "--model", OLLAMA_MODEL,
            "--prompt", prompt,
            "--stream",
            "--options", '{"num_ctx": 2048, "temperature": 0.7, "top_p": 0.9, "num_predict": 256}'
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
        self.root.title("ScreenX - AI Resume Screening")
        self.root.geometry("1000x700")
        self.is_processing = False
        self.resume_map = {}
        self.current_frame = None
        
        # Theme settings
        self.theme_mode = "light"  # Default theme
        self.load_user_preferences()
        
        self.setup_styles()
        self.show_welcome_page()
        
    def load_user_preferences(self):
        """Load user preferences from file if exists"""
        prefs_path = Path("user_preferences.json")
        if prefs_path.exists():
            try:
                with open(prefs_path, "r") as f:
                    prefs = json.load(f)
                    self.theme_mode = prefs.get("theme", "light")
            except:
                pass
                
    def save_user_preferences(self):
        """Save user preferences to file"""
        prefs = {
            "theme": self.theme_mode
        }
        try:
            with open("user_preferences.json", "w") as f:
                json.dump(prefs, f)
        except:
            pass

    def setup_styles(self):
        style = ttk.Style()
        
        # Define theme colors
        if self.theme_mode == "dark":
            self.bg_color = "#1e1e2e"  # Dark background
            self.chat_bg = "#2d2d3f"   # Slightly lighter for chat area
            self.accent_color = "#7289da"  # Discord-like accent
            self.text_color = "#e0e0e0"  # Light text
            self.highlight_color = "#5ccc8f"  # Green highlight
            self.secondary_bg = "#292938"  # Secondary background
            self.input_bg = "#383850"  # Input background
            self.user_msg_bg = "#444b5d"  # User message background
            self.bot_msg_bg = "#2d2d3f"  # Bot message background
        else:
            self.bg_color = "#f5f5f5"  # Light background
            self.chat_bg = "#ffffff"   # White for chat area
            self.accent_color = "#3498db"  # Blue accent
            self.text_color = "#2c3e50"  # Dark text
            self.highlight_color = "#2ecc71"  # Green highlight
            self.secondary_bg = "#e9ecef"  # Secondary background
            self.input_bg = "#ffffff"  # Input background
            self.user_msg_bg = "#e9f5fe"  # Light blue for user messages
            self.bot_msg_bg = "#f8f9fa"  # Light gray for bot messages
        
        # Configure styles
        style.configure("Header.TLabel", font=("Arial", 18, "bold"), foreground=self.text_color, background=self.bg_color)
        style.configure("Subheader.TLabel", font=("Arial", 14), foreground=self.text_color, background=self.bg_color)
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("Start.TButton", font=("Arial", 14, "bold"), padding=15)
        style.configure("Toggle.TButton", font=("Arial", 10), padding=5)
        
        # Configure frames
        style.configure("TFrame", background=self.bg_color)
        style.configure("Chat.TFrame", background=self.chat_bg)
        
        # Configure the root window
        self.root.configure(bg=self.bg_color)

    def show_welcome_page(self):
        # Clear previous frame if exists
        if self.current_frame:
            self.current_frame.destroy()
            
        # Create welcome frame
        welcome_frame = ttk.Frame(self.root, padding="30", style="TFrame")
        welcome_frame.pack(fill=tk.BOTH, expand=True)
        self.current_frame = welcome_frame
        
        # Theme toggle button
        theme_frame = ttk.Frame(welcome_frame, style="TFrame")
        theme_frame.pack(fill=tk.X, anchor="ne", pady=(0, 10))
        
        theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
        theme_text = f"{theme_icon} {self.theme_mode.capitalize()} Mode"
        
        theme_btn = ttk.Button(
            theme_frame,
            text=theme_text,
            command=self.toggle_theme,
            style="Toggle.TButton"
        )
        theme_btn.pack(side=tk.RIGHT, padx=5)
        
        # Welcome header with animation effect
        header_frame = ttk.Frame(welcome_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(20, 30))
        
        title_label = ttk.Label(
            header_frame, 
            text="Welcome to ScreenX", 
            style="Header.TLabel",
            foreground=self.accent_color
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(
            header_frame,
            text="AI-Powered Resume Screening Platform",
            style="Subheader.TLabel"
        )
        subtitle_label.pack()
        
        # Features section
        features_frame = ttk.Frame(welcome_frame, padding="20", style="TFrame")
        features_frame.pack(fill=tk.X, pady=20)
        
        features = [
            "📄 Upload and process multiple resumes at once",
            "🔍 Extract skills, experience, and education automatically",
            "🤖 Chat with AI to find the perfect candidates",
            "⭐ Rank candidates based on job requirements",
            "🔄 Handle synonyms and semantic matching"
        ]
        
        for feature in features:
            feature_label = ttk.Label(
                features_frame,
                text=feature,
                font=("Arial", 12),
                padding=(0, 5),
                foreground=self.text_color,
                background=self.bg_color
            )
            feature_label.pack(anchor="w", pady=5)
        
        # Start button
        button_frame = ttk.Frame(welcome_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=30)
        
        start_button = ttk.Button(
            button_frame,
            text="Get Started",
            command=self.transition_to_chatbot,
            style="Start.TButton"
        )
        start_button.pack(pady=10)
        
        # Add animation to the start button
        def pulse_button():
            try:
                padding = start_button.cget("padding")
                if isinstance(padding, str) and padding:
                    current_padding = int(padding.split()[0])
                else:
                    current_padding = 10
                    
                if current_padding == 10:
                    start_button.configure(padding=12)
                else:
                    start_button.configure(padding=10)
            except (ValueError, IndexError):
                # Handle any errors with padding
                pass
                
            welcome_frame.after(800, pulse_button)
            
        pulse_button()
        
    def toggle_theme(self):
        """Toggle between light and dark mode"""
        # Toggle theme
        self.theme_mode = "dark" if self.theme_mode == "light" else "light"
        self.save_user_preferences()
        
        # Update styles without recreating the UI
        self.setup_styles()
        
        # Update theme button text
        if hasattr(self, 'theme_btn'):
            theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
            self.theme_btn.configure(text=f"{theme_icon} {self.theme_mode.capitalize()}")
            
        # Apply theme to root window
        self.root.configure(bg=self.bg_color)
        
    def transition_to_chatbot(self):
        """Smooth transition from welcome page to chatbot UI"""
        # Create a temporary overlay for transition effect
        overlay = tk.Frame(self.root, bg="#ffffff")
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Fade out effect
        for alpha in range(10, -1, -1):
            overlay.configure(bg=f"#{int(255*alpha/10):02x}{int(255*alpha/10):02x}{int(255*alpha/10):02x}")
            self.root.update()
            time.sleep(0.03)
            
        # Switch to chatbot UI
        self.show_chatbot_ui()
        
        # Fade in effect
        for alpha in range(0, 11):
            overlay.configure(bg=f"#{int(255*alpha/10):02x}{int(255*alpha/10):02x}{int(255*alpha/10):02x}")
            self.root.update()
            time.sleep(0.03)
            
        # Remove overlay
        overlay.destroy()

    def show_chatbot_ui(self):
        # Clear welcome frame
        if self.current_frame:
            self.current_frame.destroy()
            
        # Create main frame for chatbot
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.current_frame = main_frame

        # --- Header ---
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(10, 20))
        
        ttk.Label(
            header_frame, 
            text="💬 ScreenX Resume Chatbot", 
            style="Header.TLabel",
            foreground="#3498db"
        ).pack(side=tk.LEFT, pady=10)
        
        back_btn = ttk.Button(
            header_frame, 
            text="← Back", 
            command=self.transition_to_welcome
        )
        back_btn.pack(side=tk.RIGHT, padx=5)
        
    def transition_to_welcome(self):
        """Smooth transition from chatbot UI to welcome page"""
        # Create a temporary overlay for transition effect
        overlay = tk.Frame(self.root, bg="#ffffff")
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Fade out effect
        for alpha in range(10, -1, -1):
            overlay.configure(bg=f"#{int(255*alpha/10):02x}{int(255*alpha/10):02x}{int(255*alpha/10):02x}")
            self.root.update()
            time.sleep(0.03)
            
        # Switch to welcome page
        self.show_welcome_page()
        
        # Fade in effect
        for alpha in range(0, 11):
            overlay.configure(bg=f"#{int(255*alpha/10):02x}{int(255*alpha/10):02x}{int(255*alpha/10):02x}")
            self.root.update()
            time.sleep(0.03)
            
        # Remove overlay
        overlay.destroy()

    def show_chatbot_ui(self):
        # Clear welcome frame
        if self.current_frame:
            self.current_frame.destroy()
            
        # Create main frame for chatbot
        main_frame = ttk.Frame(self.root, padding="0", style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.current_frame = main_frame

        # --- Sidebar ---
        sidebar_width = 250
        sidebar = ttk.Frame(main_frame, width=sidebar_width, style="TFrame")
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking
        
        # Logo and title in sidebar
        logo_frame = ttk.Frame(sidebar, style="TFrame")
        logo_frame.pack(fill=tk.X, pady=(20, 10), padx=15)
        
        ttk.Label(
            logo_frame, 
            text="ScreenX", 
            style="Header.TLabel",
            foreground=self.accent_color
        ).pack(side=tk.LEFT)
        
        # Theme toggle in sidebar
        theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
        theme_btn = ttk.Button(
            sidebar,
            text=f"{theme_icon} {self.theme_mode.capitalize()}",
            command=self.toggle_theme,
            style="Toggle.TButton"
        )
        theme_btn.pack(fill=tk.X, padx=15, pady=(5, 20))
        
        # Load resumes button in sidebar
        self.load_btn = ttk.Button(
            sidebar, 
            text="📁 Load Resumes Folder", 
            command=self.load_resumes,
            style="TButton"
        )
        self.load_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Back to welcome button
        back_btn = ttk.Button(
            sidebar, 
            text="← Back to Welcome", 
            command=self.transition_to_welcome,
            style="TButton"
        )
        back_btn.pack(fill=tk.X, padx=15, pady=5)
        
        # Status label in sidebar
        self.status_label = ttk.Label(
            sidebar, 
            text="Ready", 
            foreground="gray",
            background=self.bg_color
        )
        self.status_label.pack(padx=15, pady=(10, 0), anchor="w")

        # --- Chat Area (Main Content) ---
        chat_container = ttk.Frame(main_frame, style="Chat.TFrame")
        chat_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat header
        chat_header = ttk.Frame(chat_container, style="Chat.TFrame")
        chat_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        ttk.Label(
            chat_header,
            text="Chat with Resumes",
            style="Subheader.TLabel",
            foreground=self.text_color,
            background=self.chat_bg
        ).pack(anchor="w")

        # Chat messages area with improved styling
        chat_frame = ttk.Frame(chat_container, style="Chat.TFrame")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat_area = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state="disabled",
            font=("Arial", 14),
            bg=self.chat_bg,
            fg=self.text_color,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.chat_area.tag_config("user", foreground=self.text_color, font=("Arial", 14, "bold"), lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_area.tag_config("bot", foreground=self.accent_color, font=("Arial", 14), lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_area.tag_config("system", foreground="gray", font=("Arial", 12, "italic"), lmargin1=20, lmargin2=20, rmargin=20)
        self.chat_area.tag_config("user_bubble", background=self.user_msg_bg)
        self.chat_area.tag_config("bot_bubble", background=self.bot_msg_bg)
        self.chat_area.tag_config("spacing", spacing1=10, spacing3=10)
        self.chat_area.pack(fill=tk.BOTH, expand=True)

        # Input area with modern styling
        input_frame = ttk.Frame(chat_container, style="Chat.TFrame", padding=(0, 10, 0, 20))
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        input_container = ttk.Frame(input_frame, style="Chat.TFrame")
        input_container.pack(fill=tk.X)
        
        # Create a border effect for the input field
        input_border = ttk.Frame(input_container, style="Chat.TFrame")
        input_border.pack(fill=tk.X, padx=(0, 10))
        
        self.user_input = tk.Entry(
            input_border, 
            font=("Arial", 14),
            bg=self.input_bg,
            fg=self.text_color,
            insertbackground=self.text_color,  # Cursor color
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.accent_color,
            highlightcolor=self.accent_color
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=10)
        self.user_input.bind("<Return>", lambda e: self.send_question())

        self.send_btn = ttk.Button(
            input_container, 
            text="Send",
            command=self.send_question,
            style="TButton"
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Disable until loaded
        self.user_input.config(state="disabled")
        self.send_btn.config(state="disabled")
        
        # Add welcome message
        self.append_message("system", "Welcome to ScreenX! Load a folder with resumes to get started.")
        self.append_message("system", "You can ask questions like:")
        self.append_message("system", "• Who has experience with Python?")
        self.append_message("system", "• Which candidates know React?")
        self.append_message("system", "• Find candidates with AWS experience")

    def load_resumes(self):
        if self.is_processing:
            return
            
        # Use askopenfilenames instead of askdirectory to show PDF files
        pdf_files = filedialog.askopenfilenames(
            title="Select Resume PDF Files",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        
        if not pdf_files:
            return
            
        # Convert tuple of strings to list of Path objects
        pdf_files = [Path(f) for f in pdf_files]
        
        self.is_processing = True
        self.load_btn.config(state="disabled")
        self.status_label.config(text="Processing resumes...", foreground="#e74c3c")
        
        # Show loading animation
        self.show_loading_animation()
        
        threading.Thread(target=self._process_resumes, args=(pdf_files,), daemon=True).start()
        
    def show_loading_animation(self):
        """Show a simple loading animation in the chat area"""
        if not hasattr(self, 'loading_dots'):
            self.loading_dots = 0
            
        if self.is_processing:
            self.chat_area.config(state="normal")
            
            # Clear previous loading message if exists
            if hasattr(self, 'loading_message_pos'):
                self.chat_area.delete(self.loading_message_pos, f"{self.loading_message_pos} lineend")
                
            # Create loading message with dots animation
            dots = "." * (self.loading_dots % 4)
            self.loading_message_pos = self.chat_area.index("end-1c linestart")
            self.chat_area.insert(self.loading_message_pos, f"Processing resumes{dots}", "system")
            
            self.loading_dots += 1
            self.chat_area.config(state="disabled")
            self.chat_area.see("end")
            
            # Schedule next animation frame
            self.root.after(500, self.show_loading_animation)

    def _process_resumes(self, pdf_files):
        global resumes_text, candidate_names
        full_text = []
        candidate_names = []
        self.resume_map = {}
        self.pdf_paths = {}  # Store paths to original PDF files

        for i, pdf in enumerate(pdf_files):
            try:
                text = extract_text_from_pdf(str(pdf))
                name = text.splitlines()[0].strip() if text else pdf.stem
                candidate_names.append(name)
                self.resume_map[name] = text
                self.pdf_paths[name] = str(pdf)  # Store the path to the PDF file
                full_text.append(f"--- Resume: {name} ({pdf.name}) ---\n{text}")
            except Exception as e:
                full_text.append(f"[Error reading {pdf.name}: {e}]")

            self.root.after(0, lambda i=i: self.status_label.config(
                text=f"Processed {i + 1}/{len(pdf_files)} resumes...",
                foreground="#f39c12"))

        resumes_text = "\n\n".join(full_text)
        self.root.after(0, self.on_resumes_loaded)

    def on_resumes_loaded(self):
        self.is_processing = False
        self.load_btn.config(state="normal")
        self.user_input.config(state="normal")
        self.send_btn.config(state="normal")
        
        self.status_label.config(text=f"✅ Loaded {len(self.resume_map)} resumes.", foreground="#2ecc71")
        
        # Clear chat area and show welcome message
        self.chat_area.config(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.config(state="disabled")
        
        self.append_message("🤖 Bot", "Hello! I'm your resume assistant. Ask me things like:\n\n"
                                      "• Who knows Python?\n"
                                      "• Show me candidates with AWS experience\n"
                                      "• Rank these candidates for a Data Scientist role\n"
                                      "• Which candidate has the most relevant experience?\n"
                                      "• Give me John's resume (to open the PDF file)")

    def send_question(self):
        question = self.user_input.get().strip()
        if not question:
            return
        self.append_message("You", question)
        self.user_input.delete(0, tk.END)
        self.send_btn.config(state="disabled")
        
        # Show typing indicator
        self.chat_area.config(state="normal")
        self.typing_indicator_pos = self.chat_area.index("end-1c")
        self.chat_area.insert(self.typing_indicator_pos, "\n🤖 Bot is typing...", "system")
        self.chat_area.config(state="disabled")
        self.chat_area.see(tk.END)
        
        # Synchronous thread call — no asyncio!
        threading.Thread(target=self._query_sync, args=(question,), daemon=True).start()

    def _query_sync(self, question):
        """Synchronous version of query — uses local Ollama via HTTP or CLI fallback."""
        # Remove typing indicator
        self.chat_area.config(state="normal")
        self.chat_area.delete(f"{self.typing_indicator_pos} linestart", f"{self.typing_indicator_pos} lineend+1c")
        self.chat_area.config(state="disabled")
        
        # Check if this is a PDF request
        from pdf_request_handler import detect_pdf_request, extract_candidate_name, find_best_matching_candidate, open_pdf_file
        
        if detect_pdf_request(question):
            candidate_query = extract_candidate_name(question)
            if candidate_query and hasattr(self, 'pdf_paths') and self.pdf_paths:
                best_match = find_best_matching_candidate(candidate_query, list(self.pdf_paths.keys()))
                if best_match:
                    pdf_path = self.pdf_paths.get(best_match)
                    if pdf_path:
                        success = open_pdf_file(pdf_path)
                        if success:
                            self.append_message("Opening PDF file for " + best_match)
                            return
                        else:
                            self.append_message(f"Error: Could not open PDF file for {best_match}")
                            return
                self.append_message(f"Sorry, I couldn't find a resume for '{candidate_query}'")
                return
        
        # Optimize prompt for faster processing
        prompt = f"""
You are an AI assistant that answers questions based ONLY on the following resume data.

<context>
{resumes_text}
</context>

Question: {question}

Instructions:
- Answer clearly and concisely.
- If info missing, say "Not enough information".
- Keep under 3 sentences.
"""

        def token_callback(token):
            self.root.after(0, lambda: self.append_token(token))

        # This function now handles both HTTP and CLI fallback synchronously
        query_ollama_local_fallback(prompt, token_callback)

        self.root.after(0, lambda: self.send_btn.config(state="normal"))

    def append_message(self, sender: str, msg: str):
        self.chat_area.config(state="normal")
        
        # Add some spacing between messages
        if self.chat_area.get("1.0", "end-1c").strip():
            self.chat_area.insert(tk.END, "\n\n")
            
        # Add message with appropriate styling
        if sender == "You":
            self.chat_area.insert(tk.END, f"{sender}: ", "user")
            self.chat_area.insert(tk.END, msg)
        else:
            self.chat_area.insert(tk.END, f"{sender}: ", "bot")
            self.chat_area.insert(tk.END, msg)
            
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
