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
import sys

# Add CustomTkinter path to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "uilearn"))
try:
    from CustomTkinter.customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkEntry, CTkScrollableFrame, CTkTextbox
    from CustomTkinter.customtkinter import set_appearance_mode, set_default_color_theme
    CUSTOM_TKINTER_AVAILABLE = True
except ImportError:
    CUSTOM_TKINTER_AVAILABLE = False

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
            "model": "llama3.2",
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
            "--model", "llama3.2",
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

# ADD THESE NEW IMPORTS AT THE TOP
from collaboration_client import CollaborationClient
import collaboration_server
from shared_database import shared_db  # NEW: Import the shared database

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
        # Prevent window resizing
        self.root.resizable(False, False)
        self.is_processing = False
        self.resume_map = {}
        self.current_frame = None
        
        # ADD COLLABORATION CLIENT INITIALIZATION
        self.collaboration_client = CollaborationClient()
        self.collaboration_enabled = False
        
        # NEW: Database update thread
        self.db_update_thread = None
        self.db_running = False
        
        # Theme settings
        self.theme_mode = "light"  # Default theme
        self.load_user_preferences()
        
        self.setup_styles()
        self.show_welcome_page()
        self.resumes_loaded = False
        
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
        
        # Define theme colors with improved contrast
        if self.theme_mode == "dark":
            self.bg_color = "#1e1e2e"  # Dark background
            self.chat_bg = "#2d2d3f"   # Slightly lighter for chat area
            self.accent_color = "#7289da"  # Discord-like accent
            self.text_color = "#ffffff"  # Brighter text for better contrast
            self.highlight_color = "#5ccc8f"  # Green highlight
            self.secondary_bg = "#292938"  # Secondary background
            self.input_bg = "#383850"  # Input background
            self.user_msg_bg = "#444b5d"  # User message background
            self.bot_msg_bg = "#323248"  # Darker bot message background for better contrast
        else:
            self.bg_color = "#f5f5f5"  # Light background
            self.chat_bg = "#ffffff"   # White for chat area
            self.accent_color = "#2980b9"  # Darker blue accent for better contrast
            self.text_color = "#1a1a1a"  # Darker text for better contrast
            self.highlight_color = "#27ae60"  # Darker green highlight
            self.secondary_bg = "#e9ecef"  # Secondary background
            self.input_bg = "#ffffff"  # Input background
            self.user_msg_bg = "#d4e6f1"  # Darker blue for user messages (better contrast)
            self.bot_msg_bg = "#eaecee"  # Slightly darker gray for bot messages
        
        # Configure styles
        style.configure("Header.TLabel", font=("Arial", 18, "bold"), foreground=self.text_color, background=self.bg_color)
        
        # Set CustomTkinter appearance mode if available
        if CUSTOM_TKINTER_AVAILABLE:
            set_appearance_mode("dark" if self.theme_mode == "dark" else "light")
            set_default_color_theme("blue")
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
            
        # Create welcome frame with CustomTkinter if available
        if CUSTOM_TKINTER_AVAILABLE:
            # Set appearance mode based on current theme
            set_appearance_mode("dark" if self.theme_mode == "dark" else "light")
            set_default_color_theme("blue")
            
            welcome_frame = CTkFrame(self.root, corner_radius=0)
            welcome_frame.pack(fill=tk.BOTH, expand=True)
            self.current_frame = welcome_frame
            
            # Theme toggle button
            theme_frame = CTkFrame(welcome_frame, fg_color="transparent")
            theme_frame.pack(fill=tk.X, anchor="ne", pady=(10, 0), padx=20)
            
            theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
            
            self.theme_btn = CTkButton(
                theme_frame,
                text=f"{theme_icon} {self.theme_mode.capitalize()} Mode",
                command=self.toggle_theme,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8,
                width=120,
                height=32
            )
            self.theme_btn.pack(side=tk.RIGHT, padx=5)
            
            # Welcome header with animation effect
            header_frame = CTkFrame(welcome_frame, fg_color="transparent")
            header_frame.pack(fill=tk.X, pady=(30, 20), padx=40)
            
            # Logo/Icon
            logo_label = CTkLabel(
                header_frame,
                text="📄",
                font=("Arial", 48),
                text_color=self.accent_color
            )
            logo_label.pack(pady=(0, 10))
            
            title_label = CTkLabel(
                header_frame, 
                text="Welcome to ScreenX", 
                font=("Arial", 32, "bold"),
                text_color=self.accent_color
            )
            title_label.pack(pady=(0, 10))
            
            subtitle_label = CTkLabel(
                header_frame,
                text="AI-Powered Resume Screening Platform",
                font=("Arial", 18),
                text_color=self.text_color
            )
            subtitle_label.pack()
            
            # Feature cards in a grid layout
            features_frame = CTkFrame(welcome_frame, fg_color="transparent")
            features_frame.pack(fill=tk.X, pady=30, padx=40)
            
            # Configure grid
            features_frame.columnconfigure(0, weight=1)
            features_frame.columnconfigure(1, weight=1)
            features_frame.rowconfigure(0, weight=1)
            features_frame.rowconfigure(1, weight=1)
            
            # Feature 1: Load Resumes
            feature1 = CTkFrame(features_frame, corner_radius=10)
            feature1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            
            CTkLabel(
                feature1, 
                text="📁", 
                font=("Arial", 24),
                text_color=self.accent_color
            ).pack(pady=(15, 5))
            
            CTkLabel(
                feature1, 
                text="Load Resumes", 
                font=("Arial", 16, "bold"),
                text_color=self.text_color
            ).pack(pady=5)
            
            CTkLabel(
                feature1, 
                text="Import PDF resumes for AI analysis", 
                font=("Arial", 12),
                text_color=self.text_color
            ).pack(pady=(0, 10))
            
            CTkButton(
                feature1,
                text="Upload Files",
                command=self.load_resumes,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8
            ).pack(pady=(5, 15))
            
            # Feature 2: Chat with AI
            feature2 = CTkFrame(features_frame, corner_radius=10)
            feature2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
            
            CTkLabel(
                feature2, 
                text="💬", 
                font=("Arial", 24),
                text_color=self.accent_color
            ).pack(pady=(15, 5))
            
            CTkLabel(
                feature2, 
                text="Chat with AI", 
                font=("Arial", 16, "bold"),
                text_color=self.text_color
            ).pack(pady=5)
            
            CTkLabel(
                feature2, 
                text="Ask questions about candidate skills", 
                font=("Arial", 12),
                text_color=self.text_color
            ).pack(pady=(0, 10))
            
            CTkButton(
                feature2,
                text="Open Chatbot",
                command=self.transition_to_chatbot,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8
            ).pack(pady=(5, 15))
            
            # Feature 3: Team Collaboration
            feature3 = CTkFrame(features_frame, corner_radius=10)
            feature3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
            
            CTkLabel(
                feature3, 
                text="👥", 
                font=("Arial", 24),
                text_color=self.accent_color
            ).pack(pady=(15, 5))
            
            CTkLabel(
                feature3, 
                text="Team Collaboration", 
                font=("Arial", 16, "bold"),
                text_color=self.text_color
            ).pack(pady=5)
            
            CTkLabel(
                feature3, 
                text="Share insights with your team", 
                font=("Arial", 12),
                text_color=self.text_color
            ).pack(pady=(0, 10))
            
            CTkButton(
                feature3,
                text="Collaborate",
                command=self.toggle_collaboration,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8
            ).pack(pady=(5, 15))
            
            # Feature 4: Candidate Pool
            feature4 = CTkFrame(features_frame, corner_radius=10)
            feature4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
            
            CTkLabel(
                feature4, 
                text="👤", 
                font=("Arial", 24),
                text_color=self.accent_color
            ).pack(pady=(15, 5))
            
            CTkLabel(
                feature4, 
                text="Candidate Pool", 
                font=("Arial", 16, "bold"),
                text_color=self.text_color
            ).pack(pady=5)
            
            CTkLabel(
                feature4, 
                text="View and manage all candidates", 
                font=("Arial", 12),
                text_color=self.text_color
            ).pack(pady=(0, 10))
            
            CTkButton(
                feature4,
                text="View Candidates",
                command=self.show_candidate_pool,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8
            ).pack(pady=(5, 15))
            
            # Get Started button
            start_button = CTkButton(
                welcome_frame,
                text="Get Started",
                command=self.transition_to_chatbot,
                fg_color=self.accent_color,
                hover_color="#2980b9" if self.theme_mode == "light" else "#5865f2",
                corner_radius=8,
                font=("Arial", 16, "bold"),
                height=50,
                width=200
            )
            start_button.pack(pady=30)
            
            # Add animation to the start button
            def pulse_button():
                try:
                    current_width = start_button.cget("width")
                    if current_width == 200:
                        start_button.configure(width=210)
                    else:
                        start_button.configure(width=200)
                except:
                    pass
                    
                welcome_frame.after(800, pulse_button)
                
            pulse_button()
            
        else:
            # Fallback to ttk implementation
            welcome_frame = ttk.Frame(self.root, padding="30", style="TFrame")
            welcome_frame.pack(fill=tk.BOTH, expand=True)
            self.current_frame = welcome_frame
            
            # Theme toggle button
            theme_frame = ttk.Frame(welcome_frame, style="TFrame")
            theme_frame.pack(fill=tk.X, anchor="ne", pady=(0, 10))
            
            theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
            theme_text = f"{theme_icon} {self.theme_mode.capitalize()} Mode"
            
            self.theme_btn = ttk.Button(  # Store as instance variable
                theme_frame,
                text=theme_text,
                command=self.toggle_theme,
                style="Toggle.TButton"
            )
            self.theme_btn.pack(side=tk.RIGHT, padx=5)
            
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
            
            # Buttons frame
            button_frame = ttk.Frame(welcome_frame, style="TFrame")
            button_frame.pack(fill=tk.X, pady=30)
            
            # Load Resume button
            load_button = ttk.Button(
                button_frame,
                text="Load Resumes",
                command=self.load_resumes,
                style="Start.TButton"
            )
            load_button.pack(pady=20)
            
            # Chatbot button
            chatbot_button = ttk.Button(
                button_frame,
                text="Open Chatbot",
                command=self.transition_to_chatbot,
                style="Start.TButton"
            )
            chatbot_button.pack(pady=20)
            
            # Team Collaboration button
            collaboration_button = ttk.Button(
                button_frame,
                text="Team Collaboration",
                command=self.toggle_collaboration,
                style="Start.TButton"
            )
            collaboration_button.pack(pady=20)
            
            # NEW: View Candidate Pool button
            candidate_pool_button = ttk.Button(
                button_frame,
                text="View Candidate Pool",
                command=self.show_candidate_pool,
                style="Start.TButton"
            )
            candidate_pool_button.pack(pady=20)
            
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
        
        # Update theme button text if it exists
        if hasattr(self, 'theme_btn'):
            theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
            if CUSTOM_TKINTER_AVAILABLE and isinstance(self.theme_btn, CTkButton):
                self.theme_btn.configure(text=f"{theme_icon} {self.theme_mode.capitalize()}")
                # Update CustomTkinter appearance mode
                set_appearance_mode("dark" if self.theme_mode == "dark" else "light")
            else:
                self.theme_btn.configure(text=f"{theme_icon} {self.theme_mode.capitalize()}")
            
        # Apply theme to root window
        self.root.configure(bg=self.bg_color)
        
        # If we're in candidate pool view, refresh it to apply theme
        if hasattr(self, 'candidate_tree') and self.current_frame:
            self.refresh_candidate_pool()

    # ADD THESE NEW METHODS FOR CANDIDATE POOL FUNCTIONALITY

    def show_candidate_pool(self):
        """Show the candidate pool with real-time updates"""
        # Clear previous frame if exists
        if self.current_frame:
            self.current_frame.destroy()
            
        # Use CustomTkinter if available, otherwise fallback to ttk
        if CUSTOM_TKINTER_AVAILABLE:
            # Set appearance mode based on current theme
            set_appearance_mode("dark" if self.theme_mode == "dark" else "light")
            set_default_color_theme("blue")
            
            # Create candidate pool frame
            pool_frame = CTkFrame(self.root, corner_radius=0)
            pool_frame.pack(fill=tk.BOTH, expand=True)
            self.current_frame = pool_frame
            
            # Header
            header_frame = CTkFrame(pool_frame, fg_color="transparent")
            header_frame.pack(fill=tk.X, pady=(20, 20), padx=20)
            
            CTkLabel(
                header_frame,
                text="📊 Live Candidate Pool",
                font=("Arial", 20, "bold"),
                text_color=self.accent_color
            ).pack(side=tk.LEFT)
            
            # Back button
            back_btn = CTkButton(
                header_frame,
                text="← Back to Welcome",
                command=self.show_welcome_page,
                fg_color="transparent",
                hover_color="#2c3e50",
                border_width=1,
                border_color="#95a5a6",
                corner_radius=8
            )
            back_btn.pack(side=tk.RIGHT)
            
            # Stats frame
            stats_frame = CTkFrame(pool_frame, fg_color="transparent")
            stats_frame.pack(fill=tk.X, pady=(0, 20), padx=20)
            
            self.stats_label = CTkLabel(
                stats_frame,
                text="Loading statistics...",
                font=("Arial", 14)
            )
            self.stats_label.pack(anchor="w")
            
            # Search frame
            search_frame = CTkFrame(pool_frame, fg_color="transparent")
            search_frame.pack(fill=tk.X, pady=(0, 10), padx=20)
            
            CTkLabel(
                search_frame,
                text="Search:",
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            self.search_var = tk.StringVar()
            search_entry = CTkEntry(
                search_frame,
                textvariable=self.search_var,
                width=250,
                height=32,
                corner_radius=8
            )
            search_entry.pack(side=tk.LEFT, padx=(0, 10))
            search_entry.bind('<KeyRelease>', self.search_candidates)
            
            refresh_btn = CTkButton(
                search_frame,
                text="Refresh",
                command=self.refresh_candidate_pool,
                fg_color=self.accent_color,
                hover_color="#2980b9",
                corner_radius=8,
                height=32
            )
            refresh_btn.pack(side=tk.LEFT)
            
            # Candidate list frame
            list_container = CTkFrame(pool_frame, fg_color="transparent")
            list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
            
            # Create a scrollable frame for the candidate list
            list_frame = CTkScrollableFrame(list_container, fg_color="transparent")
            list_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create table header
            header_bg = "#2c3e50" if self.theme_mode == "dark" else "#ecf0f1"
            header_fg = "white" if self.theme_mode == "dark" else "black"
            
            table_header = CTkFrame(list_frame, fg_color=header_bg, corner_radius=8)
            table_header.pack(fill=tk.X, pady=(0, 2))
            
            # Configure columns with equal weights
            table_header.grid_columnconfigure(0, weight=2)
            table_header.grid_columnconfigure(1, weight=2)
            table_header.grid_columnconfigure(2, weight=3)
            table_header.grid_columnconfigure(3, weight=2)
            
            CTkLabel(table_header, text="Name", font=("Arial", 12, "bold"), text_color=header_fg).grid(row=0, column=0, padx=10, pady=10, sticky="w")
            CTkLabel(table_header, text="Email", font=("Arial", 12, "bold"), text_color=header_fg).grid(row=0, column=1, padx=10, pady=10, sticky="w")
            CTkLabel(table_header, text="Skills", font=("Arial", 12, "bold"), text_color=header_fg).grid(row=0, column=2, padx=10, pady=10, sticky="w")
            CTkLabel(table_header, text="Last Updated", font=("Arial", 12, "bold"), text_color=header_fg).grid(row=0, column=3, padx=10, pady=10, sticky="w")
            
            # Store reference to candidate rows for updating
            self.candidate_rows_frame = list_frame
            
            # Start real-time updates
            self.start_db_updates()
            
            # Load initial data
            self.refresh_candidate_pool()
        else:
            # Fallback to ttk if CustomTkinter is not available
            pool_frame = ttk.Frame(self.root, padding="20", style="TFrame")
            pool_frame.pack(fill=tk.BOTH, expand=True)
            self.current_frame = pool_frame
            
            # Header
            header_frame = ttk.Frame(pool_frame, style="TFrame")
            header_frame.pack(fill=tk.X, pady=(0, 20))
            
            ttk.Label(
                header_frame,
                text="📊 Live Candidate Pool",
                style="Header.TLabel",
                foreground=self.accent_color
            ).pack(side=tk.LEFT)
            
            # Back button
            back_btn = ttk.Button(
                header_frame,
                text="← Back to Welcome",
                command=self.show_welcome_page,
                style="TButton"
            )
            back_btn.pack(side=tk.RIGHT)
            
            # Stats frame
            stats_frame = ttk.Frame(pool_frame, style="TFrame")
            stats_frame.pack(fill=tk.X, pady=(0, 20))
            
            self.stats_label = ttk.Label(
                stats_frame,
                text="Loading statistics...",
                style="Subheader.TLabel"
            )
            self.stats_label.pack(anchor="w")
            
            # Search frame
            search_frame = ttk.Frame(pool_frame, style="TFrame")
            search_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(
                search_frame,
                text="Search:",
                style="TLabel"
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            self.search_var = tk.StringVar()
            search_entry = ttk.Entry(
                search_frame,
                textvariable=self.search_var,
                width=30
            )
            search_entry.pack(side=tk.LEFT, padx=(0, 10))
            search_entry.bind('<KeyRelease>', self.search_candidates)
            
            refresh_btn = ttk.Button(
                search_frame,
                text="Refresh",
                command=self.refresh_candidate_pool,
                style="TButton"
            )
            refresh_btn.pack(side=tk.LEFT)
            
            # Candidate list frame
            list_frame = ttk.Frame(pool_frame, style="TFrame")
            list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            
            # Create treeview for candidates
            columns = ("Name", "Email", "Skills", "Last Updated")
            self.candidate_tree = ttk.Treeview(
                list_frame,
                columns=columns,
                show="headings",
                height=15
            )
            
            # Configure columns
            self.candidate_tree.heading("Name", text="Name")
            self.candidate_tree.heading("Email", text="Email")
            self.candidate_tree.heading("Skills", text="Skills")
            self.candidate_tree.heading("Last Updated", text="Last Updated")
            
            self.candidate_tree.column("Name", width=200)
            self.candidate_tree.column("Email", width=200)
            self.candidate_tree.column("Skills", width=300)
            self.candidate_tree.column("Last Updated", width=150)
            
            # Scrollbar for treeview
            scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.candidate_tree.yview)
            self.candidate_tree.configure(yscrollcommand=scrollbar.set)
            
            self.candidate_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Double-click to view details
            self.candidate_tree.bind("<Double-1>", self.show_candidate_details)
            
            # Start real-time updates
            self.start_db_updates()
            
            # Load initial data
            self.refresh_candidate_pool()
    
    def refresh_candidate_pool(self):
        """Refresh the candidate pool display"""
        try:
            # Get all resumes from database
            resumes = shared_db.get_all_resumes()
            stats = shared_db.get_stats()
            
            # Update stats
            self.stats_label.config(
                text=f"Total Candidates: {stats['total_resumes']} | "
                     f"Last Update: {stats['last_update'] or 'Never'}"
            )
            
            # Clear existing items
            for item in self.candidate_tree.get_children():
                self.candidate_tree.delete(item)
            
            # Add resumes to treeview
            for resume in resumes:
                skills = resume.get('skills', '')[:100] + "..." if len(resume.get('skills', '')) > 100 else resume.get('skills', '')
                last_updated = resume.get('updated_at', '')[:16] if resume.get('updated_at') else 'Unknown'
                
                self.candidate_tree.insert(
                    "", "end",
                    values=(
                        resume['name'],
                        resume.get('email', ''),
                        skills,
                        last_updated
                    )
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh candidate pool: {e}")
    
    def search_candidates(self, event=None):
        """Search candidates based on query"""
        query = self.search_var.get().strip()
        if not query:
            self.refresh_candidate_pool()
            return
            
        try:
            results = shared_db.search_resumes(query)
            
            # Clear existing items
            for item in self.candidate_tree.get_children():
                self.candidate_tree.delete(item)
            
            # Add search results
            for resume in results:
                skills = resume.get('skills', '')[:100] + "..." if len(resume.get('skills', '')) > 100 else resume.get('skills', '')
                
                self.candidate_tree.insert(
                    "", "end",
                    values=(
                        resume['name'],
                        resume.get('email', ''),
                        skills,
                        "Search Result"
                    )
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Search failed: {e}")
    
    def show_candidate_details(self, event):
        """Show detailed view of selected candidate"""
        selection = self.candidate_tree.selection()
        if not selection:
            return
            
        item = self.candidate_tree.item(selection[0])
        candidate_name = item['values'][0]
        
        # Find the resume in database
        resumes = shared_db.search_resumes(candidate_name)
        if not resumes:
            messagebox.showinfo("Not Found", f"Details not found for {candidate_name}")
            return
            
        resume = resumes[0]
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Candidate Details - {candidate_name}")
        details_window.geometry("600x400")
        details_window.transient(self.root)
        details_window.grab_set()
        
        # Details content
        content_frame = ttk.Frame(details_window, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Candidate information
        info_text = f"""
Name: {resume['name']}
Email: {resume.get('email', 'Not provided')}
Phone: {resume.get('phone', 'Not provided')}

Skills:
{resume.get('skills', 'Not extracted')}

Experience:
{resume.get('experience', 'Not extracted')}

Education:
{resume.get('education', 'Not extracted')}

PDF Path: {resume.get('pdf_path', 'Not available')}
        """
        
        details_text = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            font=("Arial", 11),
            height=20
        )
        details_text.insert("1.0", info_text)
        details_text.config(state="disabled")
        details_text.pack(fill=tk.BOTH, expand=True)
    
    def start_db_updates(self):
        """Start real-time database updates"""
        if not self.db_running:
            self.db_running = True
            self.db_update_thread = threading.Thread(target=self._db_update_worker, daemon=True)
            self.db_update_thread.start()
    
    def stop_db_updates(self):
        """Stop real-time database updates"""
        self.db_running = False
    
    def _db_update_worker(self):
        """Worker thread for real-time database updates"""
        last_count = 0
        while self.db_running:
            try:
                current_stats = shared_db.get_stats()
                current_count = current_stats['total_resumes']
                
                # Refresh if count changed
                if current_count != last_count:
                    self.root.after(0, self.refresh_candidate_pool)
                    last_count = current_count
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Database update error: {e}")
                time.sleep(5)

    def extract_skills_from_text(self, text):
        """Extract skills from resume text"""
        skills_found = []
        for skill in self.KNOWN_SKILLS:
            if skill.lower() in text.lower():
                skills_found.append(skill)
        return ", ".join(skills_found) if skills_found else "Not detected"
    
    def extract_experience_from_text(self, text):
        """Extract experience information from resume text"""
        # Simple extraction - you can enhance this
        lines = text.split('\n')
        experience_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['experience', 'work', 'job', 'position']):
                experience_lines.append(line.strip())
        return "\n".join(experience_lines[:5]) if experience_lines else "Not detected"
    
    def extract_education_from_text(self, text):
        """Extract education information from resume text"""
        # Simple extraction - you can enhance this
        lines = text.split('\n')
        education_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd']):
                education_lines.append(line.strip())
        return "\n".join(education_lines[:5]) if education_lines else "Not detected"

    # MODIFY THE _process_resumes METHOD TO UPDATE DATABASE

    def _process_resumes(self, pdf_files):
        global resumes_text, candidate_names
        full_text = []
        candidate_names = []
        self.resume_map = {}
        self.pdf_paths = {}  # Store paths to original PDF files
        self.resume_data = []  # Initialize resume data list

        for i, pdf in enumerate(pdf_files):
            try:
                text = extract_text_from_pdf(str(pdf))
                name = text.splitlines()[0].strip() if text else pdf.stem
                candidate_names.append(name)
                self.resume_map[name] = text
                self.pdf_paths[name] = str(pdf)  # Store the path to the PDF file
                self.resume_data.append({"name": name, "path": str(pdf)})  # Add to resume data
                full_text.append(f"--- Resume: {name} ({pdf.name}) ---\n{text}")
                
                # NEW: Update shared database
                resume_data = {
                    'name': name,
                    'raw_text': text,
                    'pdf_path': str(pdf),
                    'skills': self.extract_skills_from_text(text),
                    'experience': self.extract_experience_from_text(text),
                    'education': self.extract_education_from_text(text)
                }
                shared_db.add_or_update_resume(resume_data)
                
            except Exception as e:
                full_text.append(f"[Error reading {pdf.name}: {e}]")
                
        resumes_text = "\n\n".join(full_text)
        
        # Share resume data via collaboration if enabled
        if self.collaboration_enabled:
            self.share_resume_data()
        
        # Update UI on main thread
        self.root.after(0, lambda: self.on_resumes_loaded(len(pdf_files)))

    # MODIFY THE apply_collaboration_update METHOD TO UPDATE DATABASE

    def apply_collaboration_update(self, resume_data):
        """Apply received resume data update"""
        if not resume_data:
            return
            
        # Update local resume data structures
        self.resume_map = resume_data.get('resume_map', {})
        self.pdf_paths = resume_data.get('pdf_paths', {})
        
        # NEW: Also update the shared database
        for name, text in self.resume_map.items():
            resume_db_data = {
                'name': name,
                'raw_text': text,
                'pdf_path': self.pdf_paths.get(name, ''),
                'skills': self.extract_skills_from_text(text),
                'experience': self.extract_experience_from_text(text),
                'education': self.extract_education_from_text(text)
            }
            shared_db.add_or_update_resume(resume_db_data)
        
        # Update the UI to reflect new data
        if hasattr(self, 'resumes_status'):
            count = len(self.resume_map)
            self.resumes_status.config(text=f"Resumes loaded: {count}")
            
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"✅ Updated {count} resumes via collaboration.", foreground="#2ecc71")
        
        # Refresh chat area if in chatbot UI
        if hasattr(self, 'chat_area'):
            self.chat_area.config(state="normal")
            self.chat_area.delete("1.0", tk.END)
            self.chat_area.insert(tk.END, "Resume data updated via team collaboration.\n\n", "system")
            self.chat_area.config(state="disabled")
            
        self.resumes_loaded = bool(self.resume_map)

    # EXISTING COLLABORATION METHODS (keep these as before)
    
    def toggle_collaboration(self):
        """Toggle team collaboration on/off"""
        if not self.collaboration_enabled:
            self.enable_collaboration()
        else:
            self.disable_collaboration()
    
    def enable_collaboration(self):
        """Enable team collaboration features"""
        # Start local server if not already running
        collaboration_server.start_collaboration_server()
        
        # Connect client to server
        if self.collaboration_client.connect():
            self.collaboration_client.set_callback(self.handle_collaboration_update)
            self.collaboration_enabled = True
            messagebox.showinfo("Team Collaboration", "Collaboration enabled! Team members can now see resume updates in real-time.")
            
            # Share current resume data if available
            if hasattr(self, 'resume_map') and self.resume_map:
                self.share_resume_data()
        else:
            messagebox.showerror("Team Collaboration", "Failed to start collaboration service. Make sure port 8888 is available.")
    
    def disable_collaboration(self):
        """Disable team collaboration features"""
        self.collaboration_enabled = False
        self.collaboration_client.disconnect()
        messagebox.showinfo("Team Collaboration", "Collaboration disabled.")
    
    def handle_collaboration_update(self, update_type, data):
        """Handle incoming collaboration updates"""
        if update_type == 'update' and data:
            # Update local resume data with received data
            self.root.after(0, lambda: self.apply_collaboration_update(data))
        elif update_type == 'sync' and data:
            # Sync with server data
            self.root.after(0, lambda: self.apply_collaboration_update(data))
    
    def share_resume_data(self):
        """Share current resume data with team"""
        if not self.collaboration_enabled or not hasattr(self, 'resume_map'):
            return
            
        resume_data = {
            'resume_map': self.resume_map,
            'pdf_paths': getattr(self, 'pdf_paths', {}),
            'timestamp': time.time()
        }
        
        self.collaboration_client.send_update(resume_data)
        
    def transition_to_chatbot(self):
        """Smooth transition from welcome page to chatbot UI"""
        # Check if resumes are loaded before transitioning
        if not self.resumes_loaded:
            messagebox.showinfo("No Resumes", "Please load resumes first before accessing the chatbot.")
            return
            
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
        
        # Use CustomTkinter if available, otherwise fallback to ttk
        if CUSTOM_TKINTER_AVAILABLE:
            # Set appearance mode based on current theme
            set_appearance_mode("dark" if self.theme_mode == "dark" else "light")
            set_default_color_theme("blue")
            
            # Create main frame for chatbot
            main_frame = CTkFrame(self.root, corner_radius=0)
            main_frame.pack(fill=tk.BOTH, expand=True)
            self.current_frame = main_frame

            # --- Sidebar ---
            sidebar_width = 250
            sidebar = CTkFrame(main_frame, width=sidebar_width, corner_radius=0)
            sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
            sidebar.pack_propagate(False)  # Prevent the sidebar from shrinking
            
            # Logo and title in sidebar
            logo_frame = CTkFrame(sidebar, fg_color="transparent")
            logo_frame.pack(fill=tk.X, pady=(20, 10), padx=15)
            
            CTkLabel(
                logo_frame, 
                text="ScreenX",
                font=("Arial", 20, "bold"),
                text_color=self.accent_color
            ).pack(side=tk.LEFT)
            
            # Theme toggle in sidebar
            theme_icon = "🌙" if self.theme_mode == "light" else "☀️"
            self.theme_btn = CTkButton(
                sidebar,
                text=f"{theme_icon} {self.theme_mode.capitalize()}",
                command=self.toggle_theme,
                fg_color=self.accent_color,
                hover_color="#2980b9",
                corner_radius=8
            )
            self.theme_btn.pack(fill=tk.X, padx=15, pady=(5, 20))
            
            # Load resumes button in sidebar
            resume_count = len(self.resume_map) if hasattr(self, 'resume_map') and self.resume_map else 0
            self.resumes_status = CTkLabel(
                sidebar, 
                text=f"Resumes loaded: {resume_count}",
                font=("Arial", 12)
            )
            self.resumes_status.pack(fill=tk.X, padx=15, pady=5)
            
            # Back to welcome button
            back_btn = CTkButton(
                sidebar, 
                text="← Back to Welcome", 
                command=self.transition_to_welcome,
                fg_color="transparent",
                hover_color="#2c3e50",
                border_width=1,
                border_color="#95a5a6",
                corner_radius=8
            )
            back_btn.pack(fill=tk.X, padx=15, pady=5)
            
            # Status label in sidebar
            self.status_label = CTkLabel(
                sidebar, 
                text="Ready", 
                text_color="gray"
            )
            self.status_label.pack(padx=15, pady=(10, 0), anchor="w")

            # --- Chat Area (Main Content) ---
            chat_container = CTkFrame(main_frame, fg_color=self.chat_bg, corner_radius=0)
            chat_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Chat header
            chat_header = CTkFrame(chat_container, fg_color="transparent")
            chat_header.pack(fill=tk.X, padx=20, pady=(20, 10))
            
            CTkLabel(
                chat_header,
                text="Chat with Resumes",
                font=("Arial", 18, "bold"),
                text_color=self.text_color
            ).pack(anchor="w")

            # Chat messages area with improved styling
            chat_frame = CTkScrollableFrame(chat_container, fg_color="transparent")
            chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.chat_area = CTkTextbox(
                chat_frame,
                wrap="word",
                state="disabled",
                font=("Arial", 14),
                fg_color="transparent",
                text_color=self.text_color,
                corner_radius=0,
                border_width=0
            )
            self.chat_area.tag_config("user", foreground=self.text_color, lmargin1=20, lmargin2=20, rmargin=20)
            self.chat_area.tag_config("bot", foreground=self.accent_color, lmargin1=20, lmargin2=20, rmargin=20)
            self.chat_area.tag_config("system", foreground="gray", lmargin1=20, lmargin2=20, rmargin=20)
            self.chat_area.tag_config("user_bubble", background=self.user_msg_bg)
            self.chat_area.tag_config("bot_bubble", background=self.bot_msg_bg)
            self.chat_area.tag_config("spacing", spacing1=10, spacing3=10)
            self.chat_area.pack(fill=tk.BOTH, expand=True)

            # Input area with modern styling
            input_frame = CTkFrame(chat_container, fg_color="transparent")
            input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
            
            input_container = CTkFrame(input_frame, fg_color="transparent")
            input_container.pack(fill=tk.X)
            
            # Create a modern input field
            self.user_input = CTkEntry(
                input_container, 
                font=("Arial", 14),
                fg_color=self.input_bg,
                text_color=self.text_color,
                border_width=1,
                border_color=self.accent_color,
                corner_radius=8,
                height=40
            )
            self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            self.user_input.bind("<Return>", lambda e: self.send_question())

            self.send_btn = CTkButton(
                input_container, 
                text="Send",
                command=self.send_question,
                fg_color=self.accent_color,
                hover_color="#2980b9",
                corner_radius=8,
                width=80,
                height=40
            )
            self.send_btn.pack(side=tk.RIGHT)
            
        else:
            # Fallback to original ttk implementation
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
            self.theme_btn = ttk.Button(  # Store as instance variable
                sidebar,
                text=f"{theme_icon} {self.theme_mode.capitalize()}",
                command=self.toggle_theme,
                style="Toggle.TButton"
            )
            self.theme_btn.pack(fill=tk.X, padx=15, pady=(5, 20))
            
            # Load resumes button in sidebar
            resume_count = len(self.resume_map) if hasattr(self, 'resume_map') and self.resume_map else 0
            self.resumes_status = ttk.Label(
                sidebar, 
                text=f"Resumes loaded: {resume_count}",
                style="TLabel"
            )
            self.resumes_status.pack(fill=tk.X, padx=15, pady=5)
            
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
            chat_frame = CTkFrame(chat_container, fg_color=self.chat_bg)
            chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Use CTkTextbox if available, otherwise fallback to ScrolledText
            self.chat_area = CTkTextbox(
                chat_frame,
                wrap="word",
                state="disabled",
                font=("Arial", 14),
                text_color=self.text_color,
                corner_radius=8,
                border_width=0,
                padx=10,
                pady=10
            )
            # CustomTkinter's CTkTextbox doesn't support font in tag_config
            self.chat_area.tag_config("user", foreground=self.text_color, lmargin1=20, lmargin2=20, rmargin=20)
            self.chat_area.tag_config("bot", foreground=self.accent_color, lmargin1=20, lmargin2=20, rmargin=20)
            self.chat_area.tag_config("system", foreground="gray", lmargin1=20, lmargin2=20, rmargin=20)
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

        # Enable/disable based on whether resumes are loaded
        if self.resumes_loaded and hasattr(self, 'resume_map') and self.resume_map:
            self.user_input.configure(state="normal")
            if CUSTOM_TKINTER_AVAILABLE and isinstance(self.send_btn, CTkButton):
                self.send_btn.configure(state="normal")
            else:
                self.send_btn.config(state="normal")
            # Add welcome message for loaded resumes
            self.append_message("🤖 Bot", "Hello! I'm your resume assistant. Ask me things like:\n\n"
                                        "• Who knows Python?\n"
                                        "• Show me candidates with AWS experience\n"
                                        "• Rank these candidates for a Data Scientist role\n"
                                        "• Which candidate has the most relevant experience?\n"
                                        "• Give me John's resume (to open the PDF file)")
        else:
            self.user_input.config(state="disabled")
            self.send_btn.config(state="disabled")
            # Add welcome message for no resumes
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
        
        # Create a temporary status label if we're on the welcome page
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Processing resumes...", foreground="#e74c3c")
        
        # Show loading animation
        self.show_loading_animation()
        
        threading.Thread(target=self._process_resumes, args=(pdf_files,), daemon=True).start()
        
    def show_loading_animation(self):
        """Show a simple loading animation in the chat area"""
        if not hasattr(self, 'loading_dots'):
            self.loading_dots = 0
            
        if self.is_processing:
            # Skip animation if chat_area doesn't exist yet
            if not hasattr(self, 'chat_area'):
                return
                
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

    def on_resumes_loaded(self, count):
        """Update UI after resumes are processed"""
        self.resumes_loaded = True
        self.is_processing = False
        
        # Enable input fields if they exist
        if hasattr(self, 'user_input'):
            self.user_input.config(state="normal")
        if hasattr(self, 'send_btn'):
            self.send_btn.config(state="normal")
        
        # Update status if it exists
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"✅ Loaded {count} resumes.", foreground="#2ecc71")
        
        # Update resumes status label if in chatbot UI
        if hasattr(self, 'resumes_status'):
            self.resumes_status.config(text=f"Resumes loaded: {count}")
        
        # Show success message
        messagebox.showinfo("Success", f"{count} resumes loaded successfully!")
        
        # Only show welcome message if chat area exists (we're in chatbot UI)
        if hasattr(self, 'chat_area'):
            self.chat_area.config(state="normal")
            self.chat_area.delete("1.0", tk.END)
            self.chat_area.insert(tk.END, "Welcome to ScreenX! You can now ask questions about the loaded resumes.\n\n", "system")
            self.chat_area.config(state="disabled")
            
            # Only append the welcome message if we're in chatbot UI
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
        self.send_btn.configure(state="disabled")
        
        # Add a small delay before showing typing indicator
        self.root.after(1000, lambda: self.start_typing_indicator())
        
        # Synchronous thread call — no asyncio!
        threading.Thread(target=self._query_sync, args=(question,), daemon=True).start()
        
    def start_typing_indicator(self):
        # Show typing indicator
        self.chat_area.configure(state="normal")
        self.typing_indicator_pos = self.chat_area.index("end-1c")
        self.chat_area.insert(self.typing_indicator_pos, "\n\n🤖 ScreenX is typing...", "system")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def _query_sync(self, question):
        """Synchronous version of query — uses local Ollama via HTTP or CLI fallback."""
        # Remove typing indicator safely
        try:
            self.chat_area.configure(state="normal")
            if hasattr(self, 'typing_indicator_pos'):
                self.chat_area.delete(f"{self.typing_indicator_pos} linestart", f"{self.typing_indicator_pos} lineend+1c")
            self.chat_area.configure(state="disabled")
        except Exception as e:
            print(f"Error removing typing indicator: {e}")
        
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
                            self.append_message("🤖 Bot", "Opening PDF file for " + best_match)
                            return
                        else:
                            self.append_message("🤖 Bot", f"Error: Could not open PDF file for {best_match}")
                            return
                self.append_message("🤖 Bot", f"Sorry, I couldn't find a resume for '{candidate_query}'")
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

        self.root.after(0, lambda: self.send_btn.configure(state="normal"))

    def append_message(self, sender, msg):
        if not hasattr(self, 'chat_area'):
            return
            
        self.chat_area.configure(state="normal")
        
        # Remove typing indicator if present
        if hasattr(self, 'typing_indicator_pos'):
            try:
                self.chat_area.delete(self.typing_indicator_pos, tk.END)
            except:
                pass
            
        # Add some spacing between messages
        if self.chat_area.get("1.0", "end-1c").strip():
            self.chat_area.insert(tk.END, "\n\n")
            
        # Add message with ChatGPT-like styling
        if sender == "user" or sender == "You":
            # Create a user message bubble similar to ChatGPT
            self.chat_area.insert(tk.END, "You: ", "user")
            self.chat_area.insert(tk.END, msg, "user_bubble")
        elif sender == "system":
            self.chat_area.insert(tk.END, "System: ", "system")
            self.chat_area.insert(tk.END, msg, "system")
        else:
            self.chat_area.insert(tk.END, "ScreenX: ", "bot")
            self.chat_area.insert(tk.END, msg, "bot_bubble")
            
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def append_token(self, token: str):
        self.chat_area.configure(state="normal")
        if self.chat_area.get("end-2c", "end") == "\n\n":
            self.chat_area.insert(tk.END, "🤖 Bot: ", "bot")
        self.chat_area.insert(tk.END, token)
        self.chat_area.configure(state="disabled")
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