
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from PyPDF2 import PdfReader, errors as pdf_errors
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langdetect import detect as lang_detect, LangDetectException

logger = logging.getLogger(__name__)


def _safe_lang_detect(text: str) -> str:
    text = (text or "").strip()
    if len(text) < 20:
        return "unknown"
    try:
        return lang_detect(text)
    except LangDetectException:
        return "unknown"


def extract_text_from_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from a PDF using a hybrid strategy:

    1. Try PyPDF2 text extraction for each page.
    2. For pages that look "empty" (very little text), fall back to OCR
       via pdf2image + pytesseract.
    3. Detect a primary language based on the concatenated text.

    Returns:
        full_text: concatenated page texts
        meta: {
            "language": primary language guess,
            "page_count": int,
        }
    """
    all_text_pages = []
    logger.info(f"Reading PDF: {path.name}")

    try:
        reader = PdfReader(str(path))
    except pdf_errors.PdfReadError as e:
        logger.warning(f"PyPDF2 could not read {path.name}: {e!r}. Trying OCR-only mode.")
        reader = None

    # First pass: direct text extraction
    if reader is not None:
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from {path.name} page {i+1}: {e!r}")
                page_text = ""
            all_text_pages.append(page_text)

    # Fallback OCR for pages that look empty
    need_ocr = [i for i, t in enumerate(all_text_pages) if len(t.strip()) < 30]
    if reader is None or len(need_ocr) == len(all_text_pages):
        # Either the whole doc failed, or every page was empty; OCR the whole thing.
        need_ocr = list(range(len(all_text_pages) or 1))

    if need_ocr:
        try:
            # pdf2image uses 1-based page numbers
            images = convert_from_path(str(path))
            for i in need_ocr:
                if i < len(images):
                    img = images[i]
                    ocr_text = pytesseract.image_to_string(img)
                    logger.info(f"OCR text extracted for {path.name} page {i+1} (len={len(ocr_text)})")
                    if i < len(all_text_pages):
                        all_text_pages[i] = ocr_text
                    else:
                        all_text_pages.append(ocr_text)
        except Exception as e:
            logger.warning(f"OCR failed for {path.name}: {e!r}")

    full_text = "\n\n".join(all_text_pages)
    language = _safe_lang_detect(full_text)
    meta = {
        "language": language,
        "page_count": len(all_text_pages),
    }
    return full_text, meta
