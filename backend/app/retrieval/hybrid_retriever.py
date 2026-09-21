"""
Hybrid retrieval orchestrator for TrustRAG.

=============================================================================
ARCHITECTURAL DECISION: RECIPROCAL RANK FUSION (RRF)
=============================================================================
1. WHAT:
   Fusing dense vector search (ChromaDB) and sparse keyword search (BM25) using
   Reciprocal Rank Fusion (RRF):
   RRF_Score(d) = SUM_{m in {dense, sparse}} ( 1 / (k + rank_m(d)) ), where k=60.

2. WHY (vs Linear Score Combination / Averaging):
   - Dense cosine similarities are strictly bounded in [0.0, 1.0].
   - BM25 scores are positive unbounded real numbers (e.g., 0.0 to 25.0+), varying wildly
     depending on query length, term frequencies, and document length.
   - Attempting to normalize or linearly blend them (e.g. 0.5 * BM25 + 0.5 * Dense) requires
     arbitrary scaling heuristics that fail as soon as query patterns or corpus statistics shift.
   - RRF bypasses score calibration entirely by operating exclusively on ORDINAL RANKS
     (1st, 2nd, 3rd...). It is scale-invariant, robust across diverse query types, and has
     been empirically proven on TREC benchmarks to consistently outperform linear combination.

3. TRADE-OFFS:
   - RRF is agnostic to score margins (it doesn't distinguish between a document that won
     rank 1 by a landslide vs by 0.001%). This is why we immediately follow RRF with a
     Cross-Encoder precision re-ranker on the top candidate window.
=============================================================================
"""

import logging
from typing import Dict, List, Optional
from app.ingestion.models import DocumentChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Two-stage hybrid retriever:
    1. Parallel Dense (ChromaDB) + Sparse (BM25) search fused with RRF (k=60).
    2. Precision Cross-Encoder re-ranking on top candidates.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid retriever components.

        Args:
            vector_store: ChromaVectorStore instance.
            bm25_index: BM25Index instance.
            reranker: CrossEncoderReranker instance.
            rrf_k: Smoothing constant for RRF (standard default is 60).
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k

    def index_chunks(self, chunks: List[DocumentChunk], reset_vector_store: bool = False):
        """
        Populates both the dense vector store and the sparse BM25 index with chunks.
        """
        logger.info(f"Indexing {len(chunks)} chunks into HybridRetriever (ChromaDB + BM25)...")
        self.vector_store.add_chunks(chunks, reset=reset_vector_store)
        self.bm25_index.index_chunks(chunks)
        logger.info("Hybrid indexing complete.")

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        enable_reranker: bool = True,
    ) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval:
        1. Query ChromaDB for top dense candidates.
        2. Query BM25 for top sparse candidates.
        3. Merge via Reciprocal Rank Fusion (RRF).
        4. (Optional) Re-rank top candidates using Cross-Encoder.

        Args:
            query: The user query string.
            top_k: Number of final results to return.
            candidate_k: Number of candidates to pull from first-stage retrieval for RRF.
            enable_reranker: Whether to run the Cross-Encoder re-ranking stage.

        Returns:
            List of RetrievalResult objects with complete ranking diagnostics.
        """
        # 1. Parallel First-Stage Retrieval
        dense_results = self.vector_store.search(query, top_k=candidate_k)
        sparse_results = self.bm25_index.search(query, top_k=candidate_k)

        # 2. Reciprocal Rank Fusion (RRF)
        # Store metadata mapping: chunk_id -> dict with intermediate stats
        candidates_map: Dict[str, Dict] = {}

        # Process dense results
        for chunk, score, rank in dense_results:
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            candidates_map[chunk.id] = {
                "chunk": chunk,
                "dense_rank": rank,
                "dense_score": score,
                "sparse_rank": None,
                "sparse_score": None,
                "rrf_score": rrf_contrib,
            }

        # Process sparse (BM25) results
        for chunk, score, rank in sparse_results:
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            if chunk.id in candidates_map:
                candidates_map[chunk.id]["sparse_rank"] = rank
                candidates_map[chunk.id]["sparse_score"] = score
                candidates_map[chunk.id]["rrf_score"] += rrf_contrib
            else:
                candidates_map[chunk.id] = {
                    "chunk": chunk,
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": score,
                    "rrf_score": rrf_contrib,
                }

        # Sort all candidates by RRF score descending
        fused_candidates = list(candidates_map.values())
        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Candidate pool for second stage
        candidate_pool = fused_candidates[:candidate_k]
        candidate_chunks = [entry["chunk"] for entry in candidate_pool]

        # 3. Second-Stage Precision Re-Ranking
        if enable_reranker and candidate_chunks:
            rerank_results = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
            # Map chunk_id to (rerank_score, final_rank)
            rerank_map = {
                chunk.id: (score, final_rank)
                for chunk, score, final_rank in rerank_results
            }

            final_results: List[RetrievalResult] = []
            for chunk, score, final_rank in rerank_results:
                info = candidates_map[chunk.id]
                final_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        dense_rank=info["dense_rank"],
                        dense_score=info["dense_score"],
                        sparse_rank=info["sparse_rank"],
                        sparse_score=info["sparse_score"],
                        rrf_score=info["rrf_score"],
                        rerank_score=score,
                        final_rank=final_rank,
                    )
                )
            return final_results
        else:
            # Return RRF ranking directly
            final_results = []
            for rank_idx, entry in enumerate(candidate_pool[:top_k]):
                final_results.append(
                    RetrievalResult(
                        chunk=entry["chunk"],
                        dense_rank=entry["dense_rank"],
                        dense_score=entry["dense_score"],
                        sparse_rank=entry["sparse_rank"],
                        sparse_score=entry["sparse_score"],
                        rrf_score=entry["rrf_score"],
                        rerank_score=None,
                        final_rank=rank_idx + 1,
                    )
                )
            return final_results
