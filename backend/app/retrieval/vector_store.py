"""
Embedded ChromaDB vector store integration for TrustRAG.

=============================================================================
ARCHITECTURAL DECISION: CONTEXTUAL CHUNK HEADERS
=============================================================================
1. WHAT:
   Prepending an explicit provenance/identity header to each chunk before embedding:
   `[Document: <name> | File: <filename> | Page: <page_number>]\n<chunk_text>`

2. WHY (vs Raw Chunk Embedding):
   - In resumes and contracts, critical identifying entities (e.g. "Alex Chen",
     "Apex Cloud Technologies Inc.") typically appear only once on the top of Page 1.
   - Chunks on later pages (e.g. technical projects or service credit formulas) never
     repeat the entity's name.
   - An isolated query like "What distributed ML projects did Alex Chen build?" will fail
     to match Chunk 2 because the bi-encoder embedding of Chunk 2 contains 0 proximity to "Alex Chen".
   - Injecting the contextual header anchors the entity to every chunk, ensuring entity queries
     retrieve the correct candidate/contract regardless of which chunk the answer lives in.

3. TRADE-OFFS:
   - Consumes ~15-20 tokens of the model's 512-token context window per chunk.
=============================================================================
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings

from app.ingestion.models import DocumentChunk
from app.retrieval.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Local, embedded ChromaDB vector store for dense chunk retrieval.
    """

    COLLECTION_NAME = "trustrag_chunks"

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize the embedded ChromaDB client.

        Args:
            persist_directory: Local disk path for vector persistence.
            embedding_engine: EmbeddingEngine instance for dense encoding.
        """
        if persist_directory is None:
            # Default to scratch/chroma_db in the project root
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            persist_directory = root_dir / "scratch" / "chroma_db"

        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_engine = embedding_engine or EmbeddingEngine()

        logger.info(f"Initializing embedded ChromaDB at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "TrustRAG document intelligence chunks with contextual headers"},
        )

        # In-memory map from chunk_id to DocumentChunk for fast reconstruction
        self._chunk_registry: Dict[str, DocumentChunk] = {}

    @staticmethod
    def format_contextual_chunk_text(chunk: DocumentChunk) -> str:
        """
        Builds the contextual chunk string prepended with document & page identity.
        """
        header = f"[Document: {chunk.document_name} | File: {chunk.source_file} | Page: {chunk.page_number}]"
        return f"{header}\n{chunk.text}"

    def add_chunks(self, chunks: List[DocumentChunk], reset: bool = False):
        """
        Embeds and stores chunks into ChromaDB with contextual headers.

        Args:
            chunks: List of DocumentChunk objects from the ingestion pipeline.
            reset: If True, clears existing records before adding.
        """
        if not chunks:
            return

        if reset:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "TrustRAG document intelligence chunks with contextual headers"},
            )
            self._chunk_registry.clear()

        # Update in-memory registry
        for c in chunks:
            self._chunk_registry[c.id] = c

        ids = [c.id for c in chunks]
        contextual_texts = [self.format_contextual_chunk_text(c) for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_engine.embed_documents(contextual_texts)

        # Metadata for ChromaDB (flat primitives only)
        metadatas = [
            {
                "source_file": c.source_file,
                "document_name": c.document_name,
                "document_id": c.document_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],  # Store raw text as document payload
            metadatas=metadatas,
        )

        logger.info(f"Successfully indexed {len(chunks)} chunks in ChromaDB collection '{self.COLLECTION_NAME}'")

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Performs semantic vector search against the collection.

        Args:
            query: Natural language query string.
            top_k: Number of candidates to return.

        Returns:
            List of tuples: (DocumentChunk, similarity_score, rank_1_indexed)
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matched_chunks: List[Tuple[DocumentChunk, float, int]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matched_chunks

        retrieved_ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(retrieved_ids)

        for rank_idx, (chunk_id, dist) in enumerate(zip(retrieved_ids, distances)):
            # Convert cosine distance to approximate similarity: sim = 1 - dist
            similarity = max(0.0, 1.0 - float(dist))

            if chunk_id in self._chunk_registry:
                chunk = self._chunk_registry[chunk_id]
            else:
                # Reconstruct from metadata if not in active registry
                meta = results["metadatas"][0][rank_idx]
                doc_text = results["documents"][0][rank_idx]
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=meta["document_id"],
                    source_file=meta["source_file"],
                    document_name=meta["document_name"],
                    chunk_index=meta["chunk_index"],
                    page_number=meta["page_number"],
                    text=doc_text,
                    token_count=meta.get("token_count", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", len(doc_text)),
                    metadata=meta,
                )
                self._chunk_registry[chunk_id] = chunk

            matched_chunks.append((chunk, similarity, rank_idx + 1))

        return matched_chunks

    def count(self) -> int:
        """Returns total number of chunks in vector store."""
        return self.collection.count()
