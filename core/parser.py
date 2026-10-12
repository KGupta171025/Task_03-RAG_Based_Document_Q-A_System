"""
Document Extraction Engine for PDF, DOCX, and TXT files.
Extracts clean plain text while maintaining page / structural provenance.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader
import docx

logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """Represents an extracted unit of text with provenance metadata."""
    text: str
    page_number: int
    source_filename: str
    file_type: str


class DocumentParser:
    """Parses PDF, DOCX, and TXT files into structured text sections."""

    @staticmethod
    def parse_pdf(file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Extracts text page-by-page from a PDF."""
        sections: List[ParsedSection] = []
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    sections.append(
                        ParsedSection(
                            text=page_text,
                            page_number=page_idx + 1,
                            source_filename=filename,
                            file_type="pdf",
                        )
                    )
        except Exception as e:
            logger.error("Failed to parse PDF '%s': %s", filename, e)
            raise ValueError(f"Could not parse PDF document: {e}")

        return sections

    @staticmethod
    def parse_docx(file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Extracts text from a DOCX document."""
        sections: List[ParsedSection] = []
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            
            # Group into synthetic pages of roughly 400 words if document is long
            words = full_text.split()
            words_per_page = 400
            if len(words) <= words_per_page:
                sections.append(
                    ParsedSection(
                        text=full_text,
                        page_number=1,
                        source_filename=filename,
                        file_type="docx",
                    )
                )
            else:
                for idx in range(0, len(words), words_per_page):
                    page_words = words[idx : idx + words_per_page]
                    page_num = (idx // words_per_page) + 1
                    sections.append(
                        ParsedSection(
                            text=" ".join(page_words),
                            page_number=page_num,
                            source_filename=filename,
                            file_type="docx",
                        )
                    )
        except Exception as e:
            logger.error("Failed to parse DOCX '%s': %s", filename, e)
            raise ValueError(f"Could not parse DOCX document: {e}")

        return sections

    @staticmethod
    def parse_txt(file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Extracts text from a TXT file with utf-8 / fallback encodings."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Could not decode text file: {e}")

        text = text.strip()
        words = text.split()
        words_per_page = 400
        sections: List[ParsedSection] = []

        if len(words) <= words_per_page:
            sections.append(
                ParsedSection(
                    text=text,
                    page_number=1,
                    source_filename=filename,
                    file_type="txt",
                )
            )
        else:
            for idx in range(0, len(words), words_per_page):
                page_words = words[idx : idx + words_per_page]
                page_num = (idx // words_per_page) + 1
                sections.append(
                    ParsedSection(
                        text=" ".join(page_words),
                        page_number=page_num,
                        source_filename=filename,
                        file_type="txt",
                    )
                )

        return sections

    @classmethod
    def parse_document(cls, file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Auto-detects document format from file extension and dispatches parser."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_bytes, filename)
        elif ext == ".docx":
            return cls.parse_docx(file_bytes, filename)
        elif ext == ".txt":
            return cls.parse_txt(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported document format '{ext}'. Must be PDF, DOCX, or TXT.")
