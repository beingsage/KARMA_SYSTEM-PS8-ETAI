import types
from io import BytesIO
import sys

from PIL import Image, ImageDraw

from app.pipeline.ocr_processor import TesseractOCRProcessor


def test_tesseract_ocr_falls_back_to_pymupdf_when_pdf2image_fails(monkeypatch):
    image = Image.new("RGB", (512, 256), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Pump valve pressure", fill="black")

    buffer = BytesIO()
    image.save(buffer, format="PDF")
    pdf_bytes = buffer.getvalue()

    fake_pdf2image = types.ModuleType("pdf2image")

    def _raise_poppler_error(_pdf_bytes):
        raise RuntimeError("Unable to get page count. Is poppler installed and in PATH?")

    fake_pdf2image.convert_from_bytes = _raise_poppler_error
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

    fake_pytesseract = types.ModuleType("pytesseract")
    fake_pytesseract.image_to_string = lambda image, lang="eng": "recognized text"
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    processor = TesseractOCRProcessor()
    result = processor.process(pdf_bytes)

    assert result["status"] == "fallback"
    assert "recognized text" in result["text"]
    assert result["processor"] in {"tesseract+pymupdf", "pymupdf-text"}
