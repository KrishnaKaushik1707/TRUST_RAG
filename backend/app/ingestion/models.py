"""
Data models for the TrustRAG ingestion pipeline.

Every document and chunk is strictly typed with provenance metadata
(source file, page numbers, character offsets) to enable precise
grounded citations and claim verification in downstream trust layers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata representing the physical source file and document attributes."""
    source_file: str = Field(..., description="Original filename with extension (e.g., 'resume_alex_chen.pdf')")
    document_name: str = Field(..., description="Human-readable document name")
    file_type: str = Field(..., description="File extension without dot ('pdf', 'docx')")
    file_size_bytes: int = Field(..., description="File size on disk in bytes")
    page_count: int = Field(default=1, description="Total number of pages/sections in document")
    ingested_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp of ingestion")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata (e.g., author, category)")


class PageContent(BaseModel):
    """Represents text extracted from a specific page/section."""
    page_number: int = Field(..., description="1-indexed physical page number")
    text: str = Field(..., description="Raw text extracted from this page")
    char_count: int = Field(..., description="Character count on this page")


class IngestedDocument(BaseModel):
    """Full document representation after parsing, holding all pages and overall text."""
    id: str = Field(..., description="Unique document ID (hash or uuid)")
    metadata: DocumentMetadata
    pages: List[PageContent] = Field(default_factory=list)
    full_text: str = Field(..., description="Aggregated text from all pages")


class DocumentChunk(BaseModel):
    """
    A discrete chunk of text ready for embedding and retrieval.
    
    Contains explicit traceability fields: page_number, char offsets,
    and chunk index, ensuring that downstream verification models can
    cite the exact location of a claim.
    """
    id: str = Field(..., description="Unique chunk ID (e.g., 'doc123_chunk_001')")
    document_id: str = Field(..., description="ID of parent IngestedDocument")
    source_file: str = Field(..., description="Source filename for direct citation")
    document_name: str = Field(..., description="Document name for user-facing citations")
    chunk_index: int = Field(..., description="0-indexed position in document sequence")
    page_number: int = Field(..., description="1-indexed page number where chunk originates")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Estimated token count (~4 chars per token)")
    char_start: int = Field(..., description="Start character offset in page text")
    char_end: int = Field(..., description="End character offset in page text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Flat dictionary ready for vector DB storage")
