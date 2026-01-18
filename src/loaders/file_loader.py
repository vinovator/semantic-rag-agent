import os
import fitz  # PyMuPDF
import csv
from typing import List

class FileLoader:
    @staticmethod
    def load_pdf(path: str) -> List[str]:
        """Extracts text from a PDF file."""
        chunks = []
        try:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text()
                if text:
                    chunks.append(text)
            doc.close()
        except Exception as e:
            print(f"Error loading PDF {path}: {e}")
        return chunks

    @staticmethod
    def load_file(path: str) -> List[str]:
        """Dispatches to the correct loader based on file extension."""
        if path.endswith(".pdf"):
            return FileLoader.load_pdf(path)
        else:
            print(f"Unsupported file type: {path}")
            return []
