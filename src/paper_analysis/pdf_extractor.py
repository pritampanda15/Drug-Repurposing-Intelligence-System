"""
PDF text extraction utilities.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files."""

    def __init__(self):
        """Initialize PDF extractor."""
        if not HAS_PDFPLUMBER and not HAS_PYPDF2:
            raise ImportError(
                "No PDF library available. Install with: pip install pdfplumber PyPDF2"
            )

        self.use_pdfplumber = HAS_PDFPLUMBER
        logger.info(f"Using {'pdfplumber' if self.use_pdfplumber else 'PyPDF2'} for PDF extraction")

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        str
            Extracted text
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.use_pdfplumber:
            return self._extract_with_pdfplumber(pdf_path)
        else:
            return self._extract_with_pypdf2(pdf_path)

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract metadata from PDF.

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        Dict[str, str]
            Metadata dictionary
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata = {}

        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                if reader.metadata:
                    metadata = {
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'producer': reader.metadata.get('/Producer', ''),
                    }

                metadata['num_pages'] = len(reader.pages)

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return metadata
