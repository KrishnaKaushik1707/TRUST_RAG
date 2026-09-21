"""
Data models for the TrustRAG retrieval subsystem.

Encapsulates individual scored chunks, intermediate rank positions
(dense, sparse, RRF), and final re-ranked outputs for full transparency.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from app.ingestion.models import DocumentChunk


class RetrievalResult(BaseModel):
    """
    Represents a retrieved document chunk with complete audit metrics:
    dense rank/score, sparse (BM25) rank/score, RRF score, and cross-encoder score.
    """
    chunk: DocumentChunk = Field(..., description="The underlying retrieved document chunk")
    
    # First-stage metrics
    dense_rank: Optional[int] = Field(None, description="Rank from ChromaDB vector search (1-indexed)")
    dense_score: Optional[float] = Field(None, description="Cosine similarity or distance from vector store")
    
    sparse_rank: Optional[int] = Field(None, description="Rank from BM25 keyword search (1-indexed)")
    sparse_score: Optional[float] = Field(None, description="BM25 raw term-frequency score")
    
    rrf_score: float = Field(0.0, description="Reciprocal Rank Fusion score combining dense + sparse")
    
    # Second-stage metrics
    rerank_score: Optional[float] = Field(None, description="Cross-encoder relevance logit/score")
    final_rank: int = Field(..., description="Final 1-indexed rank in returned result set")

    # Helper accessors
    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source_file(self) -> str:
        return self.chunk.source_file

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def document_name(self) -> str:
        return self.chunk.document_name
