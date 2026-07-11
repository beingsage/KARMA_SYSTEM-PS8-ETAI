from pathlib import Path
from typing import Optional

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()


def extract_text_from_bytes(pdf_bytes: bytes, destination: Optional[Path] = None) -> str:
    temp_path = destination or Path("/tmp/processed.pdf")
    temp_path.write_bytes(pdf_bytes)
    return extract_text_from_pdf(temp_path)
