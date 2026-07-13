"""
Industrial OCR and Document Processing Pipeline
Supports Docling primary extraction, Surya layout/table understanding, and fallback OCR.
"""

import os
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional

from PIL import Image

# Suppress expected warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Failed to load custom.*")


class BaseOCRProcessor:
    def process(self, pdf_bytes: bytes) -> Dict[str, Any]:
        raise NotImplementedError("OCR processor must implement process()")


class DoclingOCRProcessor(BaseOCRProcessor):
    """Process PDFs using Docling OCR and Surya layout/table analysis."""

    def __init__(self) -> None:
        self.processor = None
        self.surya_input = None
        self.layout_predictor = None
        self.table_predictor = None
        self.surya_ready = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.standard_pdf_pipeline import ThreadedPdfPipelineOptions
            from docling.datamodel.pipeline_options import OcrAutoOptions

            pdf_options = PdfFormatOption(
                pipeline_options=ThreadedPdfPipelineOptions(
                    do_ocr=True,
                    ocr_options=OcrAutoOptions(lang=["en"]),
                )
            )
            self.processor = DocumentConverter(format_options={InputFormat.PDF: pdf_options})
            print("✓ Docling OCR initialized for PDF OCR")
        except Exception as exc:
            print(f"✗ Docling initialization failed: {exc}")
            self.processor = None

    def _ensure_surya_ready(self) -> bool:
        if self.surya_ready:
            return True

        try:
            # Try the most common import first
            import surya.input.processing as surya_input
            layout_cls = None
            table_cls = None

            try:
                from surya.layout import LayoutPredictor
                layout_cls = LayoutPredictor
            except Exception:
                # fallback lookups: inspect surya module for layout-like classes
                try:
                    import surya as _surya
                    for name in dir(_surya):
                        if "Layout" in name or "layout" in name:
                            candidate = getattr(_surya, name)
                            if callable(candidate):
                                layout_cls = candidate
                                break
                except Exception:
                    pass

            try:
                from surya.table_rec import TableRecPredictor
                table_cls = TableRecPredictor
            except Exception:
                # best-effort fallback: try to find a table predictor in surya
                try:
                    import surya as _surya
                    for name in dir(_surya):
                        if "Table" in name or "table" in name:
                            candidate = getattr(_surya, name)
                            if callable(candidate):
                                table_cls = candidate
                                break
                except Exception:
                    pass

            if layout_cls is None or table_cls is None:
                raise ImportError("Surya API mismatch: required classes not found")

            self.surya_input = surya_input
            self.layout_predictor = layout_cls()
            self.table_predictor = table_cls()
            self.surya_ready = True
            print("✓ Surya layout and table extraction initialized")
            return True
        except Exception as exc:
            print(f"⚠ Surya initialization failed: {exc}")
            self.surya_ready = False
            return False

    def _build_reading_order(self, layout_results: List[Any]) -> List[Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        for page_index, result in enumerate(layout_results):
            if not hasattr(result, "bboxes"):
                continue
            raw_result = result.model_dump() if hasattr(result, "model_dump") else result
            boxes = [box for box in raw_result.get("bboxes", []) or []]
            def get_box_value(item: Any, key: str, default: Any = None) -> Any:
                if isinstance(item, dict):
                    return item.get(key, default)
                return getattr(item, key, default)

            boxes_sorted = sorted(
                boxes,
                key=lambda box: (
                    get_box_value(box, "bbox", [0, 0, 0, 0])[1],
                    get_box_value(box, "bbox", [0, 0, 0, 0])[0],
                ),
            )
            for position, box in enumerate(boxes_sorted):
                ordered.append(
                    {
                        "page": page_index + 1,
                        "position": position + 1,
                        "label": get_box_value(box, "label", "unknown"),
                        "confidence": float(get_box_value(box, "confidence", 0.0) or 0.0),
                        "bbox": get_box_value(box, "bbox", []),
                        "polygon": get_box_value(box, "polygon", []),
                    }
                )
        return ordered

    def _extract_formulas(self, text: str) -> List[str]:
        if not text:
            return []

        candidates: List[str] = []
        patterns = [
            r"\b[A-Z][A-Za-z]?\d+(?:[A-Z][A-Za-z]?\d+)+\b",
            r"\b[A-Za-z0-9\)\]\}]+\s*=\s*[A-Za-z0-9\(\[\{]+\b",
            r"\b[A-Za-z]+\s*\([^)]+\)\s*=\s*[A-Za-z0-9\.+\-]+\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                normalized = match.strip()
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates[:30]

    def _run_surya_analysis(self, pdf_path: str) -> Dict[str, Any]:
        if not self._ensure_surya_ready():
            return {"layout": [], "tables": [], "reading_order": []}

        try:
            doc = self.surya_input.open_pdf(pdf_path)
            page_count = len(doc)
            images = self.surya_input.get_page_images(doc, list(range(page_count)))
            layout_results = list(self.layout_predictor(images))
            table_results = list(self.table_predictor(images))

            layout = [result.model_dump() for result in layout_results]
            tables = [result.model_dump() for result in table_results]
            reading_order = self._build_reading_order(layout_results)

            return {
                "layout": layout,
                "tables": tables,
                "reading_order": reading_order,
            }
        except Exception as exc:
            print(f"⚠ Surya OCR analysis failed: {exc}")
            return {"layout": [], "tables": [], "reading_order": []}

    def process(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Docling OCR processor is not initialized.")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            path = tmp_file.name

        try:
            doc = self.processor.convert(path)
            text = doc.document.export_to_markdown()
            docling_tables = [
                table.model_dump() if hasattr(table, "model_dump") else table
                for table in getattr(doc, "tables", [])
            ]
            surya_data = self._run_surya_analysis(path)
            tables = docling_tables + surya_data.get("tables", [])
            layout = surya_data.get("layout", [])
            reading_order = surya_data.get("reading_order", [])
            formulas = self._extract_formulas(text)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        return {
            "stage": "docling_surya_ocr",
            "status": "model-backed",
            "text": text,
            "tables": tables,
            "layout": layout,
            "reading_order": reading_order,
            "formulas": formulas,
            "confidence": 0.95,
            "processor": "docling+surya" if self.surya_ready else "docling",
        }


class TesseractOCRProcessor(BaseOCRProcessor):
    """Fallback OCR processor using Tesseract with resilient PDF rendering."""

    def __init__(self) -> None:
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            import fitz  # noqa: F401
            return True
        except Exception:
            try:
                import pytesseract  # noqa: F401
                from pdf2image import convert_from_bytes  # noqa: F401
                return True
            except Exception:
                return False

    @staticmethod
    def _render_pdf_with_pdf2image(pdf_bytes: bytes) -> List[Image.Image]:
        from pdf2image import convert_from_bytes

        return convert_from_bytes(pdf_bytes)

    @staticmethod
    def _render_pdf_with_pymupdf(pdf_bytes: bytes) -> List[Image.Image]:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images: List[Image.Image] = []
        try:
            for page in document:
                pixmap = page.get_pixmap(dpi=200, alpha=False)
                mode = "RGBA" if pixmap.alpha else ("RGB" if pixmap.n >= 3 else "L")
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
        finally:
            document.close()
        return images

    @staticmethod
    def _extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
        try:
            import fitz
        except Exception:
            return ""

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts: List[str] = []
        try:
            for page in document:
                page_text = page.get_text("text").strip()
                if page_text:
                    texts.append(page_text)
        finally:
            document.close()
        return "\n\n".join(texts).strip()

    def process(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("Tesseract OCR fallback is not available. Install pytesseract and PyMuPDF or pdf2image.")

        import pytesseract

        images: List[Image.Image] = []
        render_backend = "pdf2image"

        try:
            images = self._render_pdf_with_pdf2image(pdf_bytes)
        except Exception as exc:
            print(f"⚠ pdf2image rendering failed: {exc}; falling back to PyMuPDF")
            render_backend = "pymupdf"
            try:
                images = self._render_pdf_with_pymupdf(pdf_bytes)
            except Exception as fitz_exc:
                extracted_text = self._extract_text_with_pymupdf(pdf_bytes)
                if extracted_text:
                    return {
                        "stage": "ocr_tesseract_fallback",
                        "status": "fallback",
                        "text": extracted_text,
                        "tables": [],
                        "layout": [],
                        "reading_order": [],
                        "formulas": [],
                        "confidence": 0.65,
                        "processor": "pymupdf-text",
                    }
                raise RuntimeError(
                    "Tesseract OCR fallback could not render PDF with pdf2image or PyMuPDF"
                ) from fitz_exc

        pages: List[str] = []
        try:
            for image in images:
                pages.append(pytesseract.image_to_string(image, lang="eng"))
        except Exception as exc:
            extracted_text = self._extract_text_with_pymupdf(pdf_bytes)
            if extracted_text:
                print(f"⚠ Tesseract OCR failed: {exc}; using PyMuPDF text extraction fallback")
                return {
                    "stage": "ocr_tesseract_fallback",
                    "status": "fallback",
                    "text": extracted_text,
                    "tables": [],
                    "layout": [],
                    "reading_order": [],
                    "formulas": [],
                    "confidence": 0.65,
                    "processor": "pymupdf-text",
                }
            raise

        text = "\n\n".join(page.strip() for page in pages if page)
        if not text:
            text = self._extract_text_with_pymupdf(pdf_bytes)
            if text:
                render_backend = f"{render_backend}+text"
        return {
            "stage": "ocr_tesseract_fallback",
            "status": "fallback",
            "text": text,
            "tables": [],
            "layout": [],
            "reading_order": [],
            "formulas": [],
            "confidence": 0.70,
            "processor": f"tesseract+{render_backend}",
        }


def get_best_ocr_processor() -> BaseOCRProcessor:
    try:
        processor = DoclingOCRProcessor()
        if processor.processor is not None:
            return processor
    except Exception:
        pass

    fallback = TesseractOCRProcessor()
    if fallback.available:
        print("✓ Tesseract OCR fallback ready")
        return fallback

    raise RuntimeError("No OCR processor is available. Install Docling or Tesseract dependencies.")
