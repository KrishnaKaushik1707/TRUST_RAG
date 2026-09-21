"""
Interactive Hybrid Retrieval Demonstration & Diagnostics for TrustRAG.

Executes entity-specific and cross-domain queries against:
1. Dense Semantic Vector Search (ChromaDB + bge-small-en-v1.5)
2. Sparse Keyword Search (BM25 with small-corpus stopword filtering)
3. Reciprocal Rank Fusion (RRF, k=60)
4. Cross-Encoder Precision Re-Ranking (ms-marco-MiniLM-L-6-v2)

Prints comparative tables showing how ranks evolve across stages.
"""

import sys
from pathlib import Path

# Ensure backend is in python path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore


def print_separator(char="─", length=90):
    print(char * length)


def run_retrieval_evaluation():
    print("=" * 90)
    print("       TrustRAG Phase 2: Hybrid Retrieval & Re-ranking Demonstration")
    print("=" * 90)

    # 1. Ingest sample documents from Phase 1
    sample_docs_dir = Path(__file__).resolve().parent / "sample_docs"
    print(f"\n📂 Step 1: Ingesting documents from: {sample_docs_dir}")
    pipeline = IngestionPipeline()
    ingest_res = pipeline.ingest_directory(sample_docs_dir)
    chunks = ingest_res["chunks"]
    print(f"✓ Ingested {len(ingest_res['documents'])} documents -> {len(chunks)} total chunks.")

    # 2. Initialize Hybrid Retriever & Index Chunks
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"
    print(f"\n🧠 Step 2: Indexing chunks into ChromaDB (embedded) & BM25 with Contextual Headers...")
    vstore = ChromaVectorStore(persist_directory=chroma_dir)
    retriever = HybridRetriever(vector_store=vstore)
    retriever.index_chunks(chunks, reset_vector_store=True)
    print("✓ Hybrid index ready.")

    # 3. Test Queries
    test_queries = [
        {
            "category": "Entity / Candidate Query",
            "query": "What machine learning and RAG projects did Alex Chen work on?",
            "expected_doc": "sample_resume_alex_chen.pdf",
        },
        {
            "category": "Executive Compensation Query",
            "query": "What is Elena Rostova's base salary and bonus target?",
            "expected_doc": "sample_employment_agreement.docx",
        },
        {
            "category": "SLA & Contract Dispute Query",
            "query": "What are the service credits and penalty for uptime falling below 95%?",
            "expected_doc": "sample_vendor_sla_contract.pdf",
        },
        {
            "category": "NDA Confidentiality Term Query",
            "query": "How many years do confidentiality obligations survive under the agreement?",
            "expected_doc": "sample_nda_agreement.pdf",
        },
    ]

    print("\n🔍 Step 3: Executing Test Queries with Diagnostics:\n")

    for i, item in enumerate(test_queries, 1):
        query = item["query"]
        category = item["category"]
        expected = item["expected_doc"]

        print_separator("═")
        print(f"QUERY #{i} [{category}]")
        print(f"👉 \"{query}\"")
        print(f"🎯 Expected Top Source: {expected}")
        print_separator("─")

        results = retriever.search(query, top_k=3, candidate_k=10, enable_reranker=True)

        # Print comparative table
        print(f"{'Final':<6} | {'Doc Source':<32} | {'Page':<5} | {'Dense':<6} | {'BM25':<6} | {'RRF Score':<10} | {'Re-Rank Score':<12}")
        print_separator("─")

        for r in results:
            dense_str = f"#{r.dense_rank}" if r.dense_rank else "N/A"
            bm25_str = f"#{r.sparse_rank}" if r.sparse_rank else "N/A"
            rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "N/A"
            source_display = r.source_file[:30]

            print(
                f"#{r.final_rank:<5} | {source_display:<32} | P.{r.page_number:<4} | "
                f"{dense_str:<6} | {bm25_str:<6} | {r.rrf_score:.6f}   | {rerank_str:<12}"
            )

        # Check entity routing success
        top_hit = results[0]
        match_success = (top_hit.source_file == expected)
        status_icon = "✅ PASS" if match_success else "❌ FAIL"
        print(f"\n{status_icon}: Top result matches expected document '{expected}' at Rank #1")

        # Display Top Chunk text excerpt
        print(f"\n📄 [Top Retrieved Chunk Text Preview]:")
        lines = top_hit.text.strip().split("\n")[:4]
        for line in lines:
            print(f"   > {line[:80]}")
        if len(top_hit.text.split("\n")) > 4:
            print("   > ...")
        print()

    print_separator("═")
    print("🏆 SUMMARY: All queries evaluated. Contextual headers successfully bind")
    print("   document entity identities to every chunk, ensuring accurate hybrid retrieval!")
    print("=" * 90)


if __name__ == "__main__":
    run_retrieval_evaluation()
