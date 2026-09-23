"""
TrustRAG Sentence-Level Claim Verification Module.
Audits generated answers against retrieved source passages using NLI (DeBERTa v3).
"""

import logging
import re
from typing import List, Optional, Tuple

import nltk
import torch
from sentence_transformers import CrossEncoder

from app.generation.models import SentenceVerification
from app.retrieval.models import RetrievalResult

logger = logging.getLogger("trustrag.verification")

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK punkt data: {e}")

from nltk.tokenize import sent_tokenize

# Thresholds for claim verification tiers
THRESHOLD_SUPPORTED = 0.70
THRESHOLD_PARTIAL = 0.35


class NLIVerifier:
    """
    Sentence-level claim verifier using Natural Language Inference (NLI).
    Evaluates whether each claim in a generated answer is logically entailed
    by the retrieved source passages.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading NLI Claim Verification model '{model_name}' on device '{self.device}'...")
        self.model = CrossEncoder(model_name, device=self.device)

        # Build index mapping for [contradiction, entailment, neutral]
        self.label_indices = {}
        for idx, label in self.model.config.id2label.items():
            norm_label = label.lower()
            if "entail" in norm_label:
                self.label_indices["entailment"] = int(idx)
            elif "contra" in norm_label:
                self.label_indices["contradiction"] = int(idx)
            elif "neut" in norm_label:
                self.label_indices["neutral"] = int(idx)

        logger.info(f"NLI label mapping configured: {self.label_indices}")

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits generated response text into clean, declarative sentences for verification.
        Filters out markdown headers, source attribution wrappers, and blockquote quotes.
        Handles edge cases like decimals ('3.5'), honorifics ('Dr. Smith'), and abbreviations.
        """
        raw_lines = text.strip().split("\n")
        sentences: List[str] = []

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            # Ignore markdown headings, divider lines, and meta footnotes
            if line.startswith("#") or line.startswith("---") or line.startswith("==="):
                continue
            if re.search(r"Additional context found in.*passages", line, re.IGNORECASE):
                continue
            if re.search(r"Based on \*\*.*\*\* \[Source:", line, re.IGNORECASE):
                continue

            # Clean quote markers, list bullet prefixes (- , * , 1. , etc.)
            clean_line = re.sub(r"^[>\s\-*•\d\.\)]+", "", line).strip()
            if not clean_line or len(clean_line) < 5:
                continue

            # Strip outer wrapping quotes if the whole line is a blockquote
            if clean_line.startswith('"') and clean_line.endswith('"') and len(clean_line) > 2:
                clean_line = clean_line[1:-1].strip()

            try:
                line_sents = sent_tokenize(clean_line)
            except Exception:
                # Fallback heuristic splitter
                line_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_line) if s.strip()]

            for s in line_sents:
                s_clean = s.strip()
                # Skip trivial or empty strings
                if len(s_clean) > 8 and not s_clean.startswith("[Source:"):
                    sentences.append(s_clean)

        return sentences

    def _compute_nli_probabilities(self, pairs: List[Tuple[str, str]]) -> List[dict]:
        """
        Runs batch prediction through the NLI model and returns softmax probabilities.
        """
        if not pairs:
            return []

        logits = self.model.predict(pairs)
        if len(pairs) == 1 and len(logits.shape) == 1:
            logits = [logits]

        probs_tensor = torch.softmax(torch.tensor(logits), dim=-1)
        results = []

        ent_idx = self.label_indices.get("entailment", 1)
        neu_idx = self.label_indices.get("neutral", 2)
        con_idx = self.label_indices.get("contradiction", 0)

        for p in probs_tensor:
            results.append(
                {
                    "entailment": float(p[ent_idx]),
                    "neutral": float(p[neu_idx]),
                    "contradiction": float(p[con_idx]),
                }
            )
        return results

    def verify_answer(
        self,
        answer: str,
        retrieved_chunks: List[RetrievalResult],
    ) -> List[SentenceVerification]:
        """
        Verifies every claim sentence in the generated answer against the retrieved passages.

        Args:
            answer: The complete text response from the LLM or fallback generator.
            retrieved_chunks: The top retrieved chunks used as the premise pool.

        Returns:
            List of SentenceVerification objects with confidence scores and attributions.
        """
        # If the answer is an explicit abstention refusal, no factual claims need verification
        if "I do not have enough information" in answer or not retrieved_chunks:
            return []

        sentences = self.split_sentences(answer)
        if not sentences:
            return []

        verified_results: List[SentenceVerification] = []

        for sentence in sentences:
            # 1. Build pairs: (premise = chunk_text, hypothesis = sentence)
            pairs = [(chunk.text, sentence) for chunk in retrieved_chunks]
            chunk_probs = self._compute_nli_probabilities(pairs)

            # 2. Find best supporting chunk based on entailment probability
            best_chunk_idx = -1
            best_entailment = -1.0
            best_probs = {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

            for idx, probs in enumerate(chunk_probs):
                if probs["entailment"] > best_entailment:
                    best_entailment = probs["entailment"]
                    best_chunk_idx = idx
                    best_probs = probs

            # 3. Multi-Chunk Synthesis Check:
            # If the best single chunk only reaches partial support, check if combining
            # the top-2 ranked chunks provides full entailment.
            if best_entailment < THRESHOLD_SUPPORTED and len(retrieved_chunks) >= 2:
                combined_premise = (
                    f"{retrieved_chunks[0].text[:800]}\n\n{retrieved_chunks[1].text[:800]}"
                )
                comb_probs = self._compute_nli_probabilities([(combined_premise, sentence)])[0]
                if comb_probs["entailment"] > best_entailment:
                    best_entailment = comb_probs["entailment"]
                    best_probs = comb_probs
                    best_chunk_idx = 0  # attribute primary reference to top rank

            # 4. Classify Verification Tier
            if best_entailment >= THRESHOLD_SUPPORTED:
                label = "SUPPORTED"
            elif best_entailment >= THRESHOLD_PARTIAL and best_probs["contradiction"] < 0.40:
                label = "PARTIALLY_SUPPORTED"
            else:
                label = "UNVERIFIED"

            # 5. Extract snippet and metadata from supporting chunk
            supporting_chunk = (
                retrieved_chunks[best_chunk_idx] if best_chunk_idx >= 0 else None
            )

            snippet = None
            if supporting_chunk:
                # First 200 characters of the supporting passage as preview
                snippet = supporting_chunk.text.strip().replace("\n", " ")[:200]

            verified_results.append(
                SentenceVerification(
                    sentence=sentence,
                    label=label,
                    entailment_score=round(best_probs["entailment"], 4),
                    neutral_score=round(best_probs["neutral"], 4),
                    contradiction_score=round(best_probs["contradiction"], 4),
                    supporting_chunk_id=supporting_chunk.chunk.id if supporting_chunk else None,
                    supporting_source_file=supporting_chunk.source_file if supporting_chunk else None,
                    supporting_document_name=supporting_chunk.document_name if supporting_chunk else None,
                    supporting_page_number=supporting_chunk.page_number if supporting_chunk else None,
                    supporting_text_snippet=snippet,
                )
            )

        return verified_results
