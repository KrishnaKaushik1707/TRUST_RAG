"""
Embedding engine for TrustRAG using Sentence-Transformers.

=============================================================================
ARCHITECTURAL DECISION: EMBEDDING MODEL SELECTION
=============================================================================
1. WHAT:
   Local bi-encoder embedding using `BAAI/bge-small-en-v1.5` (or `all-MiniLM-L6-v2`),
   producing 384-dimensional dense vectors.

2. WHY (vs Alternatives):
   - Alternative: 7B open models (e.g. e5-mistral-7b) or Cloud APIs (text-embedding-3-large).
   - In a 2-stage retrieval pipeline, the bi-encoder's ONLY duty is first-stage candidate
     filtering (casting a wide net for top 20-30 chunks). Fine-grained precision is handled
     by the downstream Cross-Encoder.
   - At 384 dimensions, vector generation runs in ~5ms locally on Apple Silicon / CPU, consumes
     <150MB of RAM, and ensures sensitive documents (resumes, contracts) never leave the local
     host, avoiding network latency and per-token API billing.

3. TRADE-OFFS:
   - Slightly less zero-shot nuance on complex, multi-hop reasoning queries compared to 7B models,
     which is compensated for by pairing with BM25 keyword search and a Cross-Encoder reranker.
=============================================================================
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Manages dense vector embeddings using local sentence-transformers models.
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(
                f"Failed to load primary model '{self.model_name}' ({e}). "
                f"Falling back to '{self.FALLBACK_MODEL_NAME}'..."
            )
            self.model_name = self.FALLBACK_MODEL_NAME
            self.model = SentenceTransformer(self.model_name)

        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of document chunk texts.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.
            
        Returns:
            List of float vectors, each of length self.embedding_dim.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Cosine similarity == dot product of normalized vectors
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generates a dense vector embedding for a search query.
        """
        # BGE models benefit from a query instruction if using bge-small
        query_text = query
        if "bge" in self.model_name.lower():
            query_text = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            query_text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()
