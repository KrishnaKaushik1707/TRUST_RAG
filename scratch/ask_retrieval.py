#!/usr/bin/env python3
"""
Interactive Search CLI for TrustRAG.

Type any question and watch the hybrid retriever search ChromaDB (Dense)
and BM25 (Sparse), fuse them via RRF, and re-rank with the Cross-Encoder.
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore


def main():
    print("=" * 80)
    print("        TrustRAG Interactive Hybrid Retrieval Search CLI")
    print("=" * 80)
    print("\n⏳ Ingesting sample docs and initializing hybrid index...")

    sample_docs_dir = Path(__file__).resolve().parent / "sample_docs"
    chroma_dir = Path(__file__).resolve().parent / "chroma_db"

    pipeline = IngestionPipeline()
    res = pipeline.ingest_directory(sample_docs_dir)
    chunks = res["chunks"]

    vstore = ChromaVectorStore(persist_directory=chroma_dir)
    retriever = HybridRetriever(vector_store=vstore)
    retriever.index_chunks(chunks, reset_vector_store=True)

    print(f"✓ Ready! Indexed {len(chunks)} chunks across {len(res['documents'])} documents.\n")
    print("Type any question below to test retrieval (or type 'quit' / 'exit' to stop).\n")

    while True:
        try:
            query = input("\n👉 Enter your query: ").strip()
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("Exiting search. Goodbye!")
                break

            results = retriever.search(query, top_k=3, candidate_k=10, enable_reranker=True)

            if not results:
                print("⚠️ No matching chunks found.")
                continue

            print("\n" + "─" * 80)
            print(f"{'Final':<6} | {'Doc Source':<32} | {'Page':<5} | {'Dense':<6} | {'BM25':<6} | {'Re-Rank':<10}")
            print("─" * 80)

            for r in results:
                dense_str = f"#{r.dense_rank}" if r.dense_rank else "N/A"
                bm25_str = f"#{r.sparse_rank}" if r.sparse_rank else "N/A"
                rerank_str = f"{r.rerank_score:.3f}" if r.rerank_score is not None else "N/A"
                source_display = r.source_file[:30]

                print(
                    f"#{r.final_rank:<5} | {source_display:<32} | P.{r.page_number:<4} | "
                    f"{dense_str:<6} | {bm25_str:<6} | {rerank_str:<10}"
                )

            top_chunk = results[0]
            print("\n📄 Top Match (Rank #1):")
            print(f"   • Document: {top_chunk.document_name} ({top_chunk.source_file})")
            print(f"   • Location: Page {top_chunk.page_number}, Chars [{top_chunk.chunk.char_start}:{top_chunk.chunk.char_end}]")
            print("   • Text Snippet:")
            for line in top_chunk.text.strip().split("\n")[:6]:
                print(f"     > {line[:80]}")
            if len(top_chunk.text.split("\n")) > 6:
                print("     > ...")
            print("─" * 80)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting search. Goodbye!")
            break


if __name__ == "__main__":
    main()
