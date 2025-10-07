import subprocess
import requests
import time
import os
import sys
import platform
import threading
import tkinter as tk
from tkinter import messagebox

# Default Ollama settings
OLLAMA_MODEL = "llama3.2"
OLLAMA_HTTP_URL = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/health"
MAX_STARTUP_WAIT = 5  # Further reduced wait time to 5 seconds

def is_ollama_running():
    """
    Check if the Ollama server is running by making a request to its health endpoint.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(OLLAMA_HEALTH_URL, timeout=1)  # Reduced timeout
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def is_ollama_installed():
    """
    Check if Ollama is installed on the system.
    
    Returns:
        bool: True if Ollama is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def is_model_available(model_name=OLLAMA_MODEL):
    """
    Check if the specified model is available locally.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return model_name in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def start_ollama_server():
    """
    Start the Ollama server in the background.
    
    Returns:
        subprocess.Popen: The process object for the Ollama server
    """
    # Use different commands based on the operating system
    if platform.system() == "Windows":
        # On Windows, use start command to run in background
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        process = subprocess.Popen(
            ["ollama", "serve"],
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    else:
        # On macOS/Linux
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    return process

def run_model(model_name=OLLAMA_MODEL):
    """
    Run the specified model.
    
    Args:
        model_name (str): Name of the model to run
        
    Returns:
        bool: True if model started successfully, False otherwise
    """
    try:
        subprocess.run(
            ["ollama", "run", model_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def ensure_ollama_running():
    """Ensure Ollama server is running, start it if not"""
    if is_ollama_running():
        return True
        
    if not is_ollama_installed():
        return False
        
    # Start Ollama in the background without waiting
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        return True
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False