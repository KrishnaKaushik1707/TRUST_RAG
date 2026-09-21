"""
Cross-Encoder re-ranking engine for TrustRAG.

=============================================================================
ARCHITECTURAL DECISION: TWO-STAGE RETRIEVAL WITH CROSS-ENCODER RE-RANKING
=============================================================================
1. WHAT:
   Scoring top retrieval candidates (top ~15-20 from RRF) using a cross-encoder
   model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before returning final top-k.

2. WHY (vs Bi-Encoder Alone):
   - Bi-encoders (vector models) compress an entire 400-token chunk into a single static
     vector INDEPENDENT of the query. During search, only a simple dot-product/cosine
     calculation is performed between vector centroids.
   - A Cross-Encoder feeds the (Query, Chunk) pair together into the transformer, computing
     FULL ALL-TO-ALL cross-attention across every query token and every document token
     simultaneously. It recognizes negation, modifiers, and exact phrase conditions that
     bi-encoders blur.
   - Why not cross-encode the entire vector database?
     Cross-encoders take O(N) forward passes per query—scoring 1,000 chunks would take
     seconds. But scoring the top 20 candidates takes only ~20ms on Apple Silicon/CPU!
   - This achieves the "Gold Standard" of search architecture: high-recall candidate
     generation (Dense + BM25) followed by high-precision re-ranking.

3. TRADE-OFFS:
   - Adds ~20-30ms latency per query.
   - Downloads an additional ~80MB transformer checkpoint.
=============================================================================
"""

import logging
from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder

from app.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Precision re-ranking using a cross-attention transformer.
    """

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        logger.info(f"Loading Cross-Encoder re-ranker: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Cross-Encoder re-ranker loaded successfully.")

    def rerank(
        self, query: str, candidates: List[DocumentChunk], top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Re-ranks a list of candidate chunks against a query.

        Args:
            query: The user search query.
            candidates: List of candidate DocumentChunks (typically top-15 to 20 from RRF).
            top_k: Number of highest-ranked chunks to return.

        Returns:
            List of tuples: (DocumentChunk, rerank_score, rank_1_indexed)
        """
        if not candidates:
            return []

        # Prepare (query, passage) pairs for cross-attention
        pairs = [(query, chunk.text) for chunk in candidates]

        scores = self.model.predict(pairs, show_progress_bar=False)

        # Pair candidates with their cross-encoder score
        scored_candidates = list(zip(candidates, scores))
        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank_idx, (chunk, score) in enumerate(scored_candidates[:top_k]):
            results.append((chunk, float(score), rank_idx + 1))

        return results
