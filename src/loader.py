import pymupdf as fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any


class PDFLoader:
    """Extracts text and page metadata from PDF documents using PyMuPDF."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Extracts text from each page with exact 1-indexed page numbers.
        Returns a list of dicts:
        [
            {
                "page": 1,
                "text": "...",
                "source": "filename.pdf",
                "total_pages": 10
            },
            ...
        ]
        """
        documents = []
        try:
            doc = fitz.open(str(self.file_path))
            total_pages = doc.page_count

            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text")

                # Clean basic whitespace
                cleaned_text = text.strip()
                if not cleaned_text:
                    continue  # Skip completely blank pages

                documents.append({
                    "page": page_idx + 1,  # 1-indexed page numbering
                    "text": cleaned_text,
                    "source": self.file_path.name,
                    "file_path": str(self.file_path.resolve()),
                    "total_pages": total_pages,
                })

            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF '{self.file_path.name}': {e}") from e

        if not documents:
            raise ValueError(f"No extractable text found in '{self.file_path.name}'.")

        return documents
