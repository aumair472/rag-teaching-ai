"""
PDF text extraction module.

Uses PyMuPDF (fitz) to extract page-level text with metadata.
"""

from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF documents with page-level metadata.

    Each page yields a dictionary containing the page text,
    page number, and source file name.
    """

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Absolute or relative path to the PDF.

        Returns:
            A list of dictionaries, one per page::

                {
                    "text": "...",
                    "page": 1,
                    "source_name": "lecture_notes",
                    "source_type": "pdf"
                }

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If PyMuPDF cannot open the file.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        source_name = path.stem
        pages: List[Dict[str, Any]] = []

        logger.info("Extracting PDF", extra={"source": source_name, "path": str(path)})

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise RuntimeError(f"Cannot open PDF: {exc}") from exc

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if text:  # skip blank pages
                pages.append({
                    "text": text,
                    "page": page_num + 1,  # 1-indexed
                    "source_name": source_name,
                    "source_type": "pdf",
                })

        doc.close()

        logger.info(
            "PDF extraction complete",
            extra={"source": source_name, "pages_extracted": len(pages)},
        )
        return pages
