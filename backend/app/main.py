"""
TrustRAG Backend Application Entrypoint.

Provides FastAPI REST endpoints for:
- /health: Server health and indexing diagnostics.
- /query: End-to-end hybrid retrieval, LLM synthesis, and structured citations.

CORS Configuration:
=============================================================================
Strictly defines `allow_origins` with explicit local ports:
["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]
NOTE: Under the W3C Fetch specification, setting `allow_origins=["*"]` while
`allow_credentials=True` causes browsers to silently reject responses.
Declaring explicit origins is mandatory for credentialed requests.
=============================================================================
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.generation.llm import LLMGenerator
from app.generation.models import QueryRequest, QueryResponse, RetrievedChunkInfo
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trustrag")

# Explicit CORS Origins for Vite (5173), Next/React (3000), and Docker
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:80",
    "http://127.0.0.1:80",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes vector store, indexes documents from sample_docs, and prepares the retriever.
    """
    logger.info("Initializing TrustRAG Retrieval and Generation subsystems...")

    root_dir = Path(__file__).resolve().parent.parent.parent
    sample_docs_dir = root_dir / "scratch" / "sample_docs"
    chroma_dir = root_dir / "scratch" / "chroma_db"

    # Fallback to test fixtures if sample_docs directory is empty
    if not sample_docs_dir.exists() or not list(sample_docs_dir.glob("*.pdf")):
        fixtures_dir = root_dir / "backend" / "tests" / "fixtures"
        if fixtures_dir.exists():
            sample_docs_dir = fixtures_dir

    # 1. Ingest documents
    pipeline = IngestionPipeline()
    res = pipeline.ingest_directory(sample_docs_dir)
    chunks = res["chunks"]
    app.state.total_documents = len(res["documents"])
    app.state.total_chunks = len(chunks)

    # 2. Build Hybrid Retriever
    vstore = ChromaVectorStore(persist_directory=chroma_dir)
    retriever = HybridRetriever(vector_store=vstore)
    retriever.index_chunks(chunks, reset_vector_store=True)

    app.state.retriever = retriever
    app.state.generator = LLMGenerator()

    logger.info(
        f"TrustRAG initialization complete: {app.state.total_documents} documents, "
        f"{app.state.total_chunks} chunks ready for hybrid retrieval."
    )

    yield

    logger.info("Shutting down TrustRAG...")


app = FastAPI(
    title="TrustRAG API",
    description="Trust-first Document Intelligence System with Grounded Citations & Claim Verification.",
    version="0.3.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Diagnostic health-check endpoint."""
    return {
        "status": "healthy",
        "service": "TrustRAG Backend",
        "phase": "Phase 3 - Query Endpoint & Citation UI",
        "documents_indexed": getattr(app.state, "total_documents", 0),
        "chunks_indexed": getattr(app.state, "total_chunks", 0),
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Executes end-to-end hybrid retrieval, LLM answer synthesis, and structured citation linking.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    retriever: HybridRetriever = getattr(app.state, "retriever", None)
    generator: LLMGenerator = getattr(app.state, "generator", None)

    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Retrieval engine is not initialized.")

    try:
        # 1. Two-stage hybrid retrieval (Dense + BM25 -> RRF -> Cross-Encoder)
        search_results = retriever.search(
            query=query_text,
            top_k=request.top_k,
            candidate_k=10,
            enable_reranker=True,
        )

        # 2. LLM answer generation with strict citation rules
        answer, provider, citations = generator.generate(query_text, search_results)

        # 3. Assemble diagnostic chunk info for UI inspection
        retrieved_chunk_info: List[RetrievedChunkInfo] = []
        for r in search_results:
            retrieved_chunk_info.append(
                RetrievedChunkInfo(
                    chunk_id=r.chunk.id,
                    document_name=r.document_name,
                    source_file=r.source_file,
                    page_number=r.page_number,
                    dense_rank=r.dense_rank,
                    sparse_rank=r.sparse_rank,
                    final_rank=r.final_rank,
                    rerank_score=round(r.rerank_score, 4) if r.rerank_score is not None else None,
                    text=r.text,
                )
            )

        return QueryResponse(
            query=query_text,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunk_info,
            provider=provider,
        )

    except Exception as e:
        logger.error(f"Error handling /query request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal retrieval error: {str(e)}")
