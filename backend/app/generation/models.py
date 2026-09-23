"""
Data models for the TrustRAG generation and query API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming user search query request."""
    query: str = Field(..., min_length=1, description="Natural language question or search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top retrieved passages to ground the answer")


class Citation(BaseModel):
    """Structured citation referencing a specific source passage."""
    source_file: str = Field(..., description="Source filename (e.g. 'Kaushik_ML_DEV_Resume.pdf')")
    document_name: str = Field(..., description="Human-readable document title")
    page_number: int = Field(..., description="Physical 1-indexed page number in the source file")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text_snippet: str = Field(..., description="Relevant text excerpt supporting the citation")


class RetrievedChunkInfo(BaseModel):
    """Detailed metadata and diagnostic rankings of a retrieved chunk."""
    chunk_id: str
    document_name: str
    source_file: str
    page_number: int
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    final_rank: int
    rerank_score: Optional[float] = None
    text: str


class SentenceVerification(BaseModel):
    """Sentence-level verification result with NLI probabilities and supporting source attribution."""
    sentence: str = Field(..., description="The individual claim sentence from the answer")
    label: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIED"] = Field(
        ..., description="Verification tier: SUPPORTED (green), PARTIALLY_SUPPORTED (yellow), UNVERIFIED (red)"
    )
    entailment_score: float = Field(..., description="Probability of entailment (0.0 - 1.0)")
    neutral_score: float = Field(..., description="Probability of neutral relationship (0.0 - 1.0)")
    contradiction_score: float = Field(..., description="Probability of contradiction (0.0 - 1.0)")
    supporting_chunk_id: Optional[str] = Field(None, description="Chunk ID that best entails the claim")
    supporting_source_file: Optional[str] = Field(None, description="Source filename of supporting chunk")
    supporting_document_name: Optional[str] = Field(None, description="Document title of supporting chunk")
    supporting_page_number: Optional[int] = Field(None, description="Physical page number of supporting chunk")
    supporting_text_snippet: Optional[str] = Field(None, description="Text excerpt of supporting chunk")


class QueryResponse(BaseModel):
    """Outgoing API response containing generated answer, citations, verified claims, and retrieved passages."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    verified_sentences: List[SentenceVerification] = Field(
        default_factory=list,
        description="Sentence-by-sentence claim verification analysis"
    )
    retrieved_chunks: List[RetrievedChunkInfo] = Field(default_factory=list)
    provider: str = Field(default="grounded-fallback", description="LLM provider and model used for synthesis")
