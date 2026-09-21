"""
Unit and integration tests for the TrustRAG retrieval subsystem.
"""

from pathlib import Path
import pytest
import shutil

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import ChromaVectorStore


@pytest.fixture(scope="module")
def sample_chunks():
    docs_dir = Path(__file__).resolve().parent.parent.parent / "scratch" / "sample_docs"
    pipeline = IngestionPipeline()
    res = pipeline.ingest_directory(docs_dir)
    assert len(res["chunks"]) > 0
    return res["chunks"]


@pytest.fixture
def temp_chroma_dir(tmp_path):
    chroma_dir = tmp_path / "chroma_test_db"
    yield chroma_dir
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_embedding_engine():
    engine = EmbeddingEngine()
    query_vec = engine.embed_query("Alex Chen machine learning engineer")
    assert len(query_vec) == engine.embedding_dim
    assert len(query_vec) == 384

    doc_vecs = engine.embed_documents(["First document passage.", "Second document passage."])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 384


def test_bm25_stopword_filtering():
    bm25 = BM25Index()
    raw_query = "What are the core technical skills of the candidate?"
    tokens = bm25.tokenize(raw_query, filter_stopwords=True)

    # Stopwords like "what", "are", "the", "of" should be stripped
    assert "what" not in tokens
    assert "are" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    # Informative terms remain
    assert "core" in tokens
    assert "technical" in tokens
    assert "skills" in tokens
    assert "candidate" in tokens


def test_chroma_vector_store(sample_chunks, temp_chroma_dir):
    vstore = ChromaVectorStore(persist_directory=temp_chroma_dir)
    vstore.add_chunks(sample_chunks, reset=True)
    assert vstore.count() == len(sample_chunks)

    results = vstore.search("Stripe distributed embedding inference", top_k=3)
    assert len(results) > 0
    top_chunk, score, rank = results[0]
    assert "Alex Chen" in top_chunk.document_name or "Stripe" in top_chunk.text
    assert rank == 1
    assert score > 0.0


def test_entity_retrieval_routing(sample_chunks, temp_chroma_dir):
    """
    Critical verification: tests that contextual chunk headers enable entity
    routing to return chunks from Alex Chen's resume even when querying
    achievements mentioned on sections without the candidate's explicit name.
    """
    vstore = ChromaVectorStore(persist_directory=temp_chroma_dir)
    retriever = HybridRetriever(vector_store=vstore)
    retriever.index_chunks(sample_chunks, reset_vector_store=True)

    query = "What RAG projects did Alex Chen work on?"
    results = retriever.search(query, top_k=3, enable_reranker=True)

    assert len(results) > 0
    top_result = results[0]
    # Verify the top hit is from Alex Chen's resume
    assert "sample_resume_alex_chen.pdf" == top_result.source_file
    assert top_result.final_rank == 1
    assert top_result.rerank_score is not None


def test_legal_contract_hybrid_retrieval(sample_chunks, temp_chroma_dir):
    vstore = ChromaVectorStore(persist_directory=temp_chroma_dir)
    retriever = HybridRetriever(vector_store=vstore)
    retriever.index_chunks(sample_chunks, reset_vector_store=True)

    query = "What is the 99.9% uptime commitment and service credit percentage?"
    results = retriever.search(query, top_k=3, enable_reranker=True)

    assert len(results) > 0
    top_result = results[0]
    assert "sample_vendor_sla_contract.pdf" == top_result.source_file
    assert top_result.page_number == 1
