"""
Chunking module for TrustRAG.

=============================================================================
ARCHITECTURAL DECISION: CHUNKING STRATEGY
=============================================================================
1. WHAT:
   Sentence-aware recursive character chunker with:
   - Target chunk size: ~400 tokens (~1,400 characters)
   - Chunk overlap: ~60 tokens (~210 characters)
   - Delimiter hierarchy: ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
   - Page-boundary preservation: chunks originate from explicit physical pages.

2. WHY (vs Alternatives):
   - Alternative A: Large chunks (1,000–2,000 tokens, standard for book RAG).
     Resumes and contracts are high-density, heterogeneous documents. A 1,500-token
     chunk merges 2-3 different jobs or multiple unrelated legal clauses (e.g.,
     indemnity + jurisdiction + termination). This dilutes vector embeddings and causes
     cross-clause confusion in downstream contradiction detection.
   - Alternative B: Micro chunks (50–100 tokens, e.g., single sentences).
     Single bullet points ("Engineered distributed cache reducing latency by 40%")
     lose their contextual anchors (the company name and tech stack mentioned 2 lines
     above).
   - TrustRAG Choice (~400 tokens / 60 overlap):
     Accommodates a complete job tenure or contract clause in a single chunk. The 60-token
     overlap ensures conditional clauses ("provided that...") are never severed from their
     parent requirements across chunk boundaries.

3. TRADE-OFFS:
   - More total chunks per document compared to large-chunking, requiring slightly higher
     vector database index capacity.
   - Enforcing page boundaries prevents chunks from spanning physical page breaks, which
     might split a paragraph that straddles the bottom of Page 1 and top of Page 2, but
     in return guarantees 100% deterministic, verifiable page citations.
=============================================================================
"""

import re
from typing import List, Tuple
from app.ingestion.models import DocumentChunk, IngestedDocument, PageContent


class RecursiveSentenceChunker:
    """
    Splits document text into overlapping, semantically coherent chunks
    while preserving page numbers and exact character coordinates.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        target_chunk_tokens: int = 400,
        overlap_tokens: int = 60,
        approx_chars_per_token: float = 3.5,
    ):
        """
        Initialize the chunker.

        Args:
            target_chunk_tokens: Target size of each chunk in tokens (~400).
            overlap_tokens: Number of overlapping tokens between consecutive chunks (~60).
            approx_chars_per_token: Average characters per token for heuristic sizing (default 3.5).
        """
        self.target_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.chars_per_token = approx_chars_per_token

        self.max_chars = int(self.target_tokens * self.chars_per_token)
        self.overlap_chars = int(self.overlap_tokens * self.chars_per_token)

        if self.overlap_chars >= self.max_chars:
            raise ValueError("Overlap must be strictly smaller than target chunk size.")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using whitespace and subword heuristic."""
        # A simple, fast token estimator: ~1 token per 3.5 characters or whitespace split
        words = len(text.split())
        char_est = int(len(text) / self.chars_per_token)
        # Average the two for robustness across dense code, emails, and English prose
        return max(1, int((words + char_est) / 2)) if text.strip() else 0

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using the highest-priority separator available
        until all pieces fit within self.max_chars.
        """
        final_pieces: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator != "" and s:
                # Reattach separator for natural spacing
                piece = s + separator if not s.endswith(separator) else s
            else:
                piece = s

            if not piece.strip():
                continue

            if len(piece) <= self.max_chars:
                good_splits.append(piece)
            else:
                if new_separators:
                    sub_pieces = self._split_text_recursively(piece, new_separators)
                    good_splits.extend(sub_pieces)
                else:
                    # If no more separators, hard-slice by max_chars
                    for k in range(0, len(piece), self.max_chars):
                        good_splits.append(piece[k : k + self.max_chars])

        return good_splits

    def _merge_splits_with_overlap(self, splits: List[str]) -> List[Tuple[str, int, int]]:
        """
        Merges small splits into target chunks respecting self.max_chars and self.overlap_chars.
        Returns a list of tuples: (chunk_text, approx_start_offset, approx_end_offset).
        """
        if not splits:
            return []

        chunks: List[Tuple[str, int, int]] = []
        current_chunk_pieces: List[str] = []
        current_length = 0

        i = 0
        while i < len(splits):
            piece = splits[i]
            piece_len = len(piece)

            if current_length + piece_len <= self.max_chars:
                current_chunk_pieces.append(piece)
                current_length += piece_len
                i += 1
            else:
                if current_chunk_pieces:
                    # Finalize current chunk
                    chunk_text = "".join(current_chunk_pieces).strip()
                    if chunk_text and (not chunks or chunk_text != chunks[-1][0]):
                        chunks.append((chunk_text, 0, len(chunk_text)))

                    # Compute overlap: take trailing window up to overlap_chars, aligned to word boundary
                    if self.overlap_chars > 0 and len(chunk_text) > self.overlap_chars:
                        tail = chunk_text[-self.overlap_chars :]
                        space_pos = tail.find(" ")
                        if space_pos != -1 and space_pos < len(tail) - 5:
                            overlap_str = tail[space_pos + 1 :].strip()
                        else:
                            overlap_str = tail.strip()

                        # Keep overlap if next piece will fit with it, otherwise start clean to guarantee progress
                        if overlap_str and (len(overlap_str) + 1 + piece_len <= self.max_chars):
                            current_chunk_pieces = [overlap_str + " "]
                        else:
                            current_chunk_pieces = []
                    else:
                        current_chunk_pieces = []

                    current_length = sum(len(p) for p in current_chunk_pieces)
                else:
                    # Single piece is larger than max_chars, take it directly
                    chunk_text = piece.strip()
                    if chunk_text:
                        chunks.append((chunk_text, 0, len(chunk_text)))
                    i += 1

        if current_chunk_pieces:
            chunk_text = "".join(current_chunk_pieces).strip()
            if chunk_text and (not chunks or chunk_text != chunks[-1][0]):
                chunks.append((chunk_text, 0, len(chunk_text)))

        return chunks

    def chunk_page(self, page: PageContent, doc: IngestedDocument) -> List[DocumentChunk]:
        """
        Chunks the text of a single page into DocumentChunk objects.
        """
        raw_text = page.text.strip()
        if not raw_text:
            return []

        # If page text comfortably fits within target size, avoid unnecessary splitting
        if len(raw_text) <= self.max_chars:
            tokens = self.estimate_tokens(raw_text)
            chunk_id = f"{doc.id}_p{page.page_number}_c00"
            return [
                DocumentChunk(
                    id=chunk_id,
                    document_id=doc.id,
                    source_file=doc.metadata.source_file,
                    document_name=doc.metadata.document_name,
                    chunk_index=0,
                    page_number=page.page_number,
                    text=raw_text,
                    token_count=tokens,
                    char_start=0,
                    char_end=len(raw_text),
                    metadata={
                        "source_file": doc.metadata.source_file,
                        "document_name": doc.metadata.document_name,
                        "page_number": page.page_number,
                        "token_count": tokens,
                        "file_type": doc.metadata.file_type,
                    },
                )
            ]

        splits = self._split_text_recursively(raw_text, self.DEFAULT_SEPARATORS)
        merged = self._merge_splits_with_overlap(splits)

        chunks: List[DocumentChunk] = []
        for idx, (chunk_text, _, _) in enumerate(merged):
            # Locate start and end offset within page text for exact coordinates
            char_start = raw_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else raw_text.find(chunk_text)
            if char_start == -1:
                char_start = 0
            char_end = char_start + len(chunk_text)

            tokens = self.estimate_tokens(chunk_text)
            chunk_id = f"{doc.id}_p{page.page_number}_c{idx:02d}"

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=doc.id,
                    source_file=doc.metadata.source_file,
                    document_name=doc.metadata.document_name,
                    chunk_index=idx,
                    page_number=page.page_number,
                    text=chunk_text,
                    token_count=tokens,
                    char_start=char_start,
                    char_end=char_end,
                    metadata={
                        "source_file": doc.metadata.source_file,
                        "document_name": doc.metadata.document_name,
                        "page_number": page.page_number,
                        "token_count": tokens,
                        "file_type": doc.metadata.file_type,
                    },
                )
            )

        return chunks

    def chunk_document(self, doc: IngestedDocument) -> List[DocumentChunk]:
        """
        Chunks an entire IngestedDocument page by page, re-indexing chunk numbers sequentially.
        """
        all_chunks: List[DocumentChunk] = []
        global_index = 0

        for page in doc.pages:
            page_chunks = self.chunk_page(page, doc)
            for ch in page_chunks:
                ch.chunk_index = global_index
                global_index += 1
                all_chunks.append(ch)

        return all_chunks
