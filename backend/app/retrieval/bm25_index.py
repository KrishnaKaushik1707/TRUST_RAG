"""
BM25 sparse keyword retrieval index for TrustRAG.

=============================================================================
ARCHITECTURAL DECISION: BM25 STOPWORD FILTERING ON SMALL CORPORA
=============================================================================
1. WHAT:
   Exact keyword retrieval using BM25Okapi with case-folding, regex tokenization,
   and strict English stopword filtering before computing term weights.

2. WHY (vs Unfiltered BM25):
   - BM25 calculates term importance using Inverse Document Frequency (IDF):
     IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
   - In massive search corpora (N = 100,000 documents), common English words like "what",
     "are", "the", "does" appear in 90%+ of documents. The math naturally drives their
     IDF down to ~0.
   - On a SMALL corpus (e.g. TrustRAG's 7-chunk test set or a 20-page contract), a query
     like "What are the ML projects of Alex Chen?" will find that the word "what" appears
     in only 1 or 2 chunks!
   - BM25's formula mistakenly calculates "what" as an extremely rare, high-value keyword,
     giving massive score bonuses to chunks that happen to contain the word "what" rather
     than the target entities ("Alex", "Chen", "projects", "ML").
   - Removing stopwords beforehand eliminates this mathematical artifact on small N.

3. TRADE-OFFS:
   - Idiomatic phrases composed mostly of stopwords (e.g., "to be or not to be") have
     their terms removed. We guard against this by falling back to unfiltered tokens if
     a query consists entirely of stopwords.
=============================================================================
"""

import logging
import re
from typing import List, Optional, Set, Tuple
from rank_bm25 import BM25Okapi

from app.ingestion.models import DocumentChunk
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# Standard English stopwords
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class BM25Index:
    """
    BM25Okapi sparse keyword index with small-corpus stopword filtering.
    """

    def __init__(self, stopwords: Optional[Set[str]] = None):
        self.stopwords = stopwords or DEFAULT_STOPWORDS
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def tokenize(self, text: str, filter_stopwords: bool = True) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric words, filtering stopwords.
        """
        raw_tokens = re.findall(r"\b\w+\b", text.lower())
        if filter_stopwords:
            filtered = [t for t in raw_tokens if t not in self.stopwords and len(t) > 1]
            # If all tokens were filtered out (e.g. user queried "what is it"), fallback to raw tokens
            return filtered if filtered else raw_tokens
        return raw_tokens

    def index_chunks(self, chunks: List[DocumentChunk]):
        """
        Builds the BM25 index over the provided chunks.
        Applies contextual headers so that entities (candidate/document names)
        are searchable via BM25 keywords as well.
        """
        self.chunks = list(chunks)
        self.tokenized_corpus = []

        for chunk in self.chunks:
            # Include contextual header in sparse index so entity names match
            contextual_text = ChromaVectorStore.format_contextual_chunk_text(chunk)
            tokens = self.tokenize(contextual_text, filter_stopwords=True)
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index over {len(self.chunks)} chunks.")
        else:
            self.bm25 = None

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Performs BM25 keyword search for a query.

        Args:
            query: Natural language or keyword query string.
            top_k: Max candidates to return.

        Returns:
            List of tuples: (DocumentChunk, bm25_score, rank_1_indexed)
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair chunks with scores
        scored_pairs = list(zip(self.chunks, scores))
        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank_idx, (chunk, score) in enumerate(scored_pairs[:top_k]):
            # If score is 0.0, the chunk has zero matching terms
            results.append((chunk, float(score), rank_idx + 1))

        return results
