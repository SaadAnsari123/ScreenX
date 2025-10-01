import re
import os
import subprocess
from difflib import get_close_matches

def detect_pdf_request(query):
    """
    Detect if the query is asking for a PDF file.
    
    Args:
        query (str): The user's query
        
    Returns:
        bool: True if the query is asking for a PDF, False otherwise
    """
    patterns = [
        r"(?:give|send|show|get|download|open).*(?:resume|cv|pdf).*(?:of|for)\s+(.+?)(?:\s|$|\.)",
        r"(?:give|send|show|get|download|open)\s+(.+?)(?:'s)?\s+(?:resume|cv|pdf)",
        r"(?:resume|cv|pdf).*(?:of|for)\s+(.+?)(?:\s|$|\.)",
        r"(?:.+?)(?:'s)\s+(?:resume|cv|pdf)",
        r"(?:candidate|applicant)\s+(\d+).*(?:resume|cv|pdf)",
    ]
    
    for pattern in patterns:
        if re.search(pattern, query.lower()):
            return True
    
    return False

def extract_candidate_name(query):
    """
    Extract the candidate name from a PDF request query.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: The extracted candidate name or None if not found
    """
    patterns = [
        r"(?:give|send|show|get|download|open).*(?:resume|cv|pdf).*(?:of|for)\s+(.+?)(?:\s|$|\.)",
        r"(?:give|send|show|get|download|open)\s+(.+?)(?:'s)?\s+(?:resume|cv|pdf)",
        r"(?:resume|cv|pdf).*(?:of|for)\s+(.+?)(?:\s|$|\.)",
        r"(.+?)(?:'s)\s+(?:resume|cv|pdf)",
        r"(?:candidate|applicant)\s+(\d+).*(?:resume|cv|pdf)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    
    return None

def find_best_matching_candidate(candidate_query, candidate_names):
    """
    Find the best matching candidate name from the available candidates.
    
    Args:
        candidate_query (str): The candidate name from the query
        candidate_names (list): List of available candidate names
        
    Returns:
        str: The best matching candidate name or None if no good match
    """
    # Try exact match first
    for name in candidate_names:
        if candidate_query.lower() in name.lower():
            return name
    
    # Try fuzzy matching
    matches = get_close_matches(candidate_query.lower(), 
                               [name.lower() for name in candidate_names], 
                               n=1, 
                               cutoff=0.6)
    
    if matches:
        # Find the original case-sensitive name
        for name in candidate_names:
            if name.lower() == matches[0]:
                return name
    
    return None

def open_pdf_file(pdf_path):
    """
    Open the PDF file using the default system viewer.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(pdf_path):
            if os.name == 'nt':  # Windows
                os.startfile(pdf_path)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.call(('xdg-open' if os.uname().sysname == 'Linux' else 'open', pdf_path))
            return True
        return False
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return False