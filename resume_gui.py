import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Dict
import threading
import asyncio
from resume_processor import ResumeProcessor

class ResumeProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Resume Processor")
        self.root.geometry("800x600")
        self.processor = ResumeProcessor()
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=('Arial', 10))
        self.style.configure("Header.TLabel", font=('Arial', 12, 'bold'))
        self.style.configure("Success.TFrame", background="#e6ffe6")
        self.style.configure("Error.TFrame", background="#ffe6e6")
        
        self.create_widgets()
        self.processing_queue = []
        self.active_tasks = 0
        self.max_concurrent = 2  # Process 2 resumes at once

    def create_widgets(self):
        """Initialize all GUI components"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(
            main_frame, 
            text="AI Resume Processor", 
            style="Header.TLabel"
        ).pack(pady=10)
        
        # Drop area
        self.drop_area = tk.Label(
            main_frame,
            text="Drag & Drop PDF Files Here\nor\nClick to Select Files",
            relief=tk.RIDGE,
            borderwidth=2,
            padx=50,
            pady=50,
            bg="white"
        )
        self.drop_area.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Bind events
        self.drop_area.bind("<Button-1>", self.select_files)
        self.drop_area.bind("<Enter>", lambda e: self.drop_area.config(bg="#e6f3ff"))
        self.drop_area.bind("<Leave>", lambda e: self.drop_area.config(bg="white"))
        
        # Progress container
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to process resumes")
        ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN
        ).pack(fill=tk.X, pady=(5,0))

    def select_files(self, event=None):
        """Handle file selection dialog"""
        files = filedialog.askopenfilenames(
            title="Select Resume PDFs",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if files:
            self.add_to_queue(files)

    def add_to_queue(self, files: List[str]):
        """Add files to processing queue"""
        new_files = [f for f in files if f.lower().endswith('.pdf')]
        self.processing_queue.extend(new_files)
        self.status_var.set(f"Added {len(new_files)} files. Total queue: {len(self.processing_queue)}")
        self.process_queue()

    def process_queue(self):
        """Process files from queue with concurrency control"""
        while self.processing_queue and self.active_tasks < self.max_concurrent:
            file_path = self.processing_queue.pop(0)
            self.active_tasks += 1
            self.create_progress_item(file_path)
            
            # Start processing in thread
            threading.Thread(
                target=self.process_file,
                args=(file_path,),
                daemon=True
            ).start()

    def create_progress_item(self, file_path: str):
        """Create a progress bar for a file"""
        frame = ttk.Frame(self.progress_frame)
        frame.pack(fill=tk.X, pady=2)
        
        # Filename label
        ttk.Label(frame, text=Path(file_path).name, width=40).pack(side=tk.LEFT)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        ttk.Progressbar(
            frame, 
            variable=progress_var, 
            maximum=100
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Status label
        status_var = tk.StringVar(value="Waiting")
        ttk.Label(frame, textvariable=status_var, width=15).pack(side=tk.RIGHT)
        
        # Store references
        frame.progress_var = progress_var
        frame.status_var = status_var
        frame.file_path = file_path

    def process_file(self, file_path: str):
        """Process a single file with progress updates"""
        def update_progress(p: float):
            """Update progress from 0-1"""
            for frame in self.progress_frame.winfo_children():
                if hasattr(frame, 'file_path') and frame.file_path == file_path:
                    frame.progress_var.set(p * 100)
                    frame.status_var.set(
                        "Extracting" if p < 0.1 else
                        "Processing" if p < 0.9 else
                        "Uploading" if p < 1.0 else
                        "Done"
                    )
                    if p >= 1.0:
                        frame.config(style="Success.TFrame")
                    break
        
        try:
            # Run async processing in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.processor.process_resume(
                    file_path,
                    progress_callback=update_progress
                )
            )
            
        except Exception as e:
            self.update_error_status(file_path, str(e))
        finally:
            self.active_tasks -= 1
            self.root.after(100, self.process_queue)
            loop.close()

    def update_error_status(self, file_path: str, error: str):
        """Mark a file as errored"""
        for frame in self.progress_frame.winfo_children():
            if hasattr(frame, 'file_path') and frame.file_path == file_path:
                frame.status_var.set("Error")
                frame.config(style="Error.TFrame")
                messagebox.showerror(
                    "Processing Error",
                    f"Failed to process {Path(file_path).name}:\n{error}"
                )
                break

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeProcessorGUI(root)
    root.mainloop()