"""
Document parsers for TrustRAG.

Supports:
- PDF (via PyMuPDF / fitz, with layout and page boundary preservation)
- DOCX (via python-docx, extracting both paragraphs and structured tables)
"""

from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
from typing import List, Optional

from app.ingestion.models import DocumentMetadata, IngestedDocument, PageContent


class BaseParser(ABC):
    """Abstract base class for all file parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> IngestedDocument:
        """
        Parse a file and return a structured IngestedDocument object.
        
        Args:
            file_path: Path to the target document.
            
        Returns:
            IngestedDocument with per-page text and metadata.
        """
        pass

    @staticmethod
    def generate_doc_id(file_path: Path) -> str:
        """Generate a deterministic ID based on file path and file size."""
        stat = file_path.stat()
        unique_string = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def infer_document_name(file_path: Path) -> str:
        """Infer a human-friendly document title from filename."""
        name = file_path.stem
        # Replace underscores and hyphens with spaces and title-case
        cleaned = name.replace("_", " ").replace("-", " ")
        return " ".join(word.capitalize() for word in cleaned.split())


class PDFParser(BaseParser):
    """
    High-performance PDF parser using PyMuPDF (fitz).
    
    PyMuPDF preserves reading-block order across multi-column layouts
    (common in modern resumes and dual-column contracts) and provides
    exact page-by-page text extraction.
    """

    def parse(self, file_path: Path) -> IngestedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        doc_id = self.generate_doc_id(file_path)
        doc_name = self.infer_document_name(file_path)

        pages: List[PageContent] = []
        full_text_parts: List[str] = []

        try:
            import pymupdf

            pdf_doc = pymupdf.open(file_path)
            total_pages = len(pdf_doc)

            for page_idx in range(total_pages):
                page = pdf_doc[page_idx]
                # 'text' mode preserves layout blocks in reading order
                page_text = page.get_text("text").strip()
                page_num = page_idx + 1

                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=page_text,
                        char_count=len(page_text),
                    )
                )
                if page_text:
                    full_text_parts.append(page_text)

            pdf_doc.close()

        except ImportError:
            # Fallback to pypdf if fitz is not installed
            import pypdf

            reader = pypdf.PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_text = (page.extract_text() or "").strip()
                page_num = page_idx + 1

                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=page_text,
                        char_count=len(page_text),
                    )
                )
                if page_text:
                    full_text_parts.append(page_text)

        metadata = DocumentMetadata(
            source_file=file_path.name,
            document_name=doc_name,
            file_type="pdf",
            file_size_bytes=file_size,
            page_count=max(len(pages), 1),
        )

        return IngestedDocument(
            id=doc_id,
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(full_text_parts),
        )


class DocxParser(BaseParser):
    """
    Parser for Microsoft Word (.docx) documents using python-docx.
    
    Extracts both paragraph text and tabular data (e.g. fee tables,
    SLA metrics) commonly found in enterprise contracts.
    """

    def parse(self, file_path: Path) -> IngestedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        import docx

        doc = docx.Document(str(file_path))
        file_size = file_path.stat().st_size
        doc_id = self.generate_doc_id(file_path)
        doc_name = self.infer_document_name(file_path)

        content_blocks: List[str] = []

        # 1. Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                content_blocks.append(text)

        # 2. Extract tables (format as markdown-style tables for clear semantics)
        for table in doc.tables:
            table_rows: List[str] = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Filter out redundant empty cells
                if any(cells):
                    table_rows.append(" | ".join(cells))
            if table_rows:
                content_blocks.append("\n".join(table_rows))

        full_text = "\n\n".join(content_blocks)

        # Word documents are continuous streams rather than fixed physical pages.
        # We synthesize sections or treat as Page 1 for short contracts/resumes.
        pages = [
            PageContent(
                page_number=1,
                text=full_text,
                char_count=len(full_text),
            )
        ]

        metadata = DocumentMetadata(
            source_file=file_path.name,
            document_name=doc_name,
            file_type="docx",
            file_size_bytes=file_size,
            page_count=1,
        )

        return IngestedDocument(
            id=doc_id,
            metadata=metadata,
            pages=pages,
            full_text=full_text,
        )


def get_parser_for_file(file_path: Path) -> BaseParser:
    """
    Factory function returning the appropriate parser for a given file extension.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFParser()
    elif suffix == ".docx":
        return DocxParser()
    else:
        raise ValueError(f"Unsupported document format '{suffix}'. Supported formats: .pdf, .docx")
