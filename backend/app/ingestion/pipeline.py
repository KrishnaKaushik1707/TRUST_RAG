"""
Orchestration pipeline for TrustRAG document ingestion.

Loads files from directories or individual paths, delegates parsing
to the appropriate file parser (PDF/DOCX), generates traceability
metadata, and outputs chunks ready for vector indexing in Phase 2.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.ingestion.chunker import RecursiveSentenceChunker
from app.ingestion.models import DocumentChunk, IngestedDocument
from app.ingestion.parsers import get_parser_for_file

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion orchestrator for TrustRAG.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(self, chunker: Optional[RecursiveSentenceChunker] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            chunker: Sentence-aware chunker instance. If None, default
                     ~400 tokens / 60 tokens overlap chunker is used.
        """
        self.chunker = chunker or RecursiveSentenceChunker()

    def ingest_file(self, file_path: Union[str, Path]) -> Tuple[IngestedDocument, List[DocumentChunk]]:
        """
        Ingests a single document: parses content, extracts metadata, and chunks text.

        Args:
            file_path: Path to the target document.

        Returns:
            Tuple containing:
            - IngestedDocument: Full parsed document with per-page text
            - List[DocumentChunk]: Traceable chunks with page numbers and offsets
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target document does not exist: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{path.suffix}'. Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        # 1. Parse document
        parser = get_parser_for_file(path)
        doc = parser.parse(path)

        # 2. Chunk document
        chunks = self.chunker.chunk_document(doc)

        logger.info(
            f"Successfully ingested '{doc.metadata.source_file}' "
            f"({doc.metadata.page_count} pages, {len(chunks)} chunks, {len(doc.full_text)} chars)"
        )

        return doc, chunks

    def ingest_directory(
        self, dir_path: Union[str, Path], recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Discovers and ingests all supported documents within a directory.

        Args:
            dir_path: Path to directory containing documents.
            recursive: Whether to search nested subdirectories.

        Returns:
            Dictionary containing:
            - documents: List of all IngestedDocument objects
            - chunks: List of all DocumentChunk objects across all documents
            - stats: Aggregated statistics (total docs, total pages, total chunks)
        """
        folder = Path(dir_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {folder}")

        pattern = "**/*" if recursive else "*"
        all_files = [
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        documents: List[IngestedDocument] = []
        all_chunks: List[DocumentChunk] = []

        for f in sorted(all_files):
            try:
                doc, chunks = self.ingest_file(f)
                documents.append(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error ingesting file {f.name}: {e}", exc_info=True)

        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "total_pages": sum(d.metadata.page_count for d in documents),
            "total_characters": sum(len(d.full_text) for d in documents),
            "files_processed": [d.metadata.source_file for d in documents],
        }

        return {
            "documents": documents,
            "chunks": all_chunks,
            "stats": stats,
        }
