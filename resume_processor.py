import re
import asyncio
import aiohttp
import orjson
import requests
from pathlib import Path
from datetime import datetime, timezone
import pypdfium2 as pdfium
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Optional, Callable

class ResumeProcessor:
    def __init__(self):
        self.config = {
            "OLLAMA_URL": "http://localhost:11434/api/generate",
            "OLLAMA_MODEL": "llama3",
            "OLLAMA_TIMEOUT": 30,
            "MAX_LLAMA_TRIES": 3,
            "SUPABASE_URL": "https://rawijfjlyspjmjqxoitj.supabase.co",
            "SUPABASE_KEY":  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJhd2lqZmpseXNwam1qcXhvaXRqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3MDkxNjEsImV4cCI6MjA2OTI4NTE2MX0.U7HfqWZLcp0MBTokQQp5qn9cxoodDtfdgG6u4xiwroc",
            "TABLE": "Resume_Details"
        }
        
        self.LLAMA_PROMPT = """
        Return a compact JSON object with keys: candidate_name, email, phone, 
        skills (array), experience (int), highest_edu, role_applied. Missing → null.
        Resume:
        {text}
        """

    def extract_text(self, pdf_path: str) -> str:
        """Faster PDF text extraction using pypdfium2"""
        doc = pdfium.PdfDocument(pdf_path)
        return "\n".join(page.get_textpage().get_text_bounded() for page in doc).strip()

    def extract_json(self, raw: str) -> Dict:
        """Robust JSON extraction with multiple fallbacks"""
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return orjson.loads(match.group())
            raise ValueError("No valid JSON found in response")

    def llama_extract(self, text: str, progress_callback: Optional[Callable] = None) -> Dict:
        """LLaMA processing with streaming and progress updates"""
        session = requests.Session()
        retry = Retry(
            total=self.config["MAX_LLAMA_TRIES"],
            backoff_factor=1.5,
            status_forcelist=[502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retry))

        response = session.post(
            self.config["OLLAMA_URL"],
            json={
                "model": self.config["OLLAMA_MODEL"],
                "prompt": self.LLAMA_PROMPT.format(text=text),
                "stream": True
            },
            timeout=self.config["OLLAMA_TIMEOUT"],
            stream=True
        )
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = orjson.loads(line.decode())
                    result += chunk.get("response", "")
                    if progress_callback:
                        progress_callback(len(result) / (len(text) * 2))  # Approximate progress
                except Exception:
                    continue

        return self.extract_json(result)

    async def upload_to_supabase(self, data: Dict) -> bool:
        """Async Supabase upload"""
        async with aiohttp.ClientSession(
            headers={
                "apikey": self.config["SUPABASE_KEY"],
                "Authorization": f"Bearer {self.config['SUPABASE_KEY']}",
                "Content-Type": "application/json"
            }
        ) as session:
            async with session.post(
                f"{self.config['SUPABASE_URL']}/rest/v1/{self.config['TABLE']}",
                data=orjson.dumps(data)
            ) as response:
                if response.status == 201:
                    return True
                raise RuntimeError(f"Upload failed: {await response.text()}")

    async def process_resume(
        self, 
        pdf_path: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """Complete processing pipeline"""
        # Step 1: Extract text (10% progress)
        if progress_callback: progress_callback(0.1)
        text = self.extract_text(pdf_path)

        # Step 2: Process with LLaMA (10-90% progress)
        data = self.llama_extract(text, 
            lambda p: progress_callback(0.1 + p*0.8) if progress_callback else None
        )

        # Step 3: Upload (90-100% progress)
        if progress_callback: progress_callback(0.9)
        data["screened_on"] = datetime.now(timezone.utc).isoformat()
        await self.upload_to_supabase(data)
        if progress_callback: progress_callback(1.0)

        return data