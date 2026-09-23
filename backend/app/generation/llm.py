"""
LLM Generation Engine with Source Citations and Hallucination Guardrails.

=============================================================================
ARCHITECTURAL DECISION: LLM PROVIDER & HALLUCINATION GUARDRAILS
=============================================================================
1. WHAT:
   - Provider: Groq with `llama-3.1-8b-instant` (sub-second LPU inference) with
     pluggable support for Google Gemini Flash and an offline Grounded Fallback mode.
   - Prompt: Strict "I don't know" instruction forbidding out-of-context extrapolation
     and enforcing inline bracket citations: `[Source: <filename>, Page: <p>]`.

2. WHY (vs Unconstrained / Giant Models):
   - Groq LPUs generate text at 500-800 tokens/sec. For an interactive UI and live demo,
     instant responses (<400ms) prevent sluggish UI waiting.
   - A 8B model is completely sufficient because our Phase 2 hybrid retriever already
     isolated the exact 3-5 relevant chunks (~1,500 tokens).
   - The "I don't know" instruction is the FIRST LINE OF DEFENSE against hallucinations
     in high-stakes document intelligence (hiring, legal contracts). It forms the baseline
     that will be audited by Phase 4's Claim Verification engine.

3. TRADE-OFFS:
   - Llama 3.1 8B is conservative under strict prompting and will decline to answer
     questions where facts must be loosely guessed, which is deliberate for a Trust-first system.
=============================================================================
"""

import logging
import os
import re
from typing import List, Optional, Tuple

from app.generation.models import Citation, QueryResponse, RetrievedChunkInfo
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TrustRAG, an enterprise document intelligence assistant.
Your job is to answer the user's question with 100% accuracy, strictly grounded in the provided document excerpts.

STRICT OPERATING RULES:
1. ONLY use factual statements directly supported by the context passages below.
2. If the context does not contain enough information to answer the question with certainty, respond with:
   "I do not have enough information in the provided documents to answer this question."
3. Do NOT extrapolate, speculate, or introduce external knowledge.
4. For every factual claim you make, cite the source in brackets, e.g.: [Source: <filename>, Page: <page_number>].
5. Keep your answer concise, objective, and clearly organized.
"""


class LLMGenerator:
    """
    Synthesizes answers from retrieved context passages with precise source citations.
    """

    def __init__(self, provider: Optional[str] = None):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.provider = provider or os.getenv("LLM_PROVIDER", "auto")

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Formats retrieved chunks into a numbered context block for the prompt."""
        context_blocks = []
        for i, r in enumerate(results, 1):
            block = (
                f"--- EXCERPT #{i} ---\n"
                f"Document: {r.document_name}\n"
                f"Source File: {r.source_file}\n"
                f"Page Number: {r.page_number}\n"
                f"Content:\n{r.text.strip()}\n"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def _call_groq(self, prompt: str, context: str) -> str:
        """Invokes Groq API with Llama 3.1 8B."""
        from groq import Groq

        client = Groq(api_key=self.groq_api_key)
        user_message = f"CONTEXT PASSAGES:\n{context}\n\nUSER QUESTION: {prompt}"

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # Low temperature for strict factual grounding
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini(self, prompt: str, context: str) -> str:
        """Invokes Google Gemini Flash."""
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        user_message = f"CONTEXT PASSAGES:\n{context}\n\nUSER QUESTION: {prompt}"
        response = model.generate_content(
            user_message,
            generation_config={"temperature": 0.1, "max_output_tokens": 800},
        )
        return response.text.strip()

    def _grounded_fallback(self, query: str, results: List[RetrievalResult]) -> str:
        """
        Deterministic, offline fallback mode when no API key is provided.
        Constructs a structured, cited answer directly from the top retrieved passages.
        """
        if not results:
            return "I do not have enough information in the provided documents to answer this question."

        top_hit = results[0]
        # Clean excerpt text
        snippet_lines = [line.strip() for line in top_hit.text.strip().split("\n") if line.strip()]
        preview = " ".join(snippet_lines[:3])

        answer = (
            f"Based on **{top_hit.document_name}** [Source: {top_hit.source_file}, Page: {top_hit.page_number}]:\n\n"
            f"> \"{preview}\"\n\n"
            f"*(Additional context found in {len(results)} matching passages across your documents)*."
        )
        return answer

    def generate(
        self, query: str, results: List[RetrievalResult]
    ) -> Tuple[str, str, List[Citation]]:
        """
        Generates a grounded answer with citations from the retrieved results.

        Returns:
            Tuple of (generated_answer, active_provider, list_of_citations)
        """
        if not results:
            return (
                "I do not have enough information in the provided documents to answer this question.",
                "none",
                [],
            )

        context_str = self._format_context(results)
        answer = ""
        active_provider = "grounded-fallback"

        # Try Groq if configured
        if self.groq_api_key and (self.provider in {"auto", "groq"}):
            try:
                answer = self._call_groq(query, context_str)
                active_provider = "groq/llama-3.1-8b-instant"
            except Exception as e:
                logger.error(f"Groq API call failed: {e}. Falling back...", exc_info=True)

        # Try Gemini if Groq was not used and Gemini is configured
        if not answer and self.gemini_api_key and (self.provider in {"auto", "gemini"}):
            try:
                answer = self._call_gemini(query, context_str)
                active_provider = "gemini-1.5-flash"
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}. Falling back...", exc_info=True)

        # Fallback to direct citation synthesis if no LLM API available
        if not answer:
            answer = self._grounded_fallback(query, results)
            active_provider = "grounded-fallback (no API key configured)"

        # Build structured citations from the retrieved passages
        citations: List[Citation] = []
        for r in results:
            first_line = r.text.strip().split("\n")[0][:120]
            citations.append(
                Citation(
                    source_file=r.source_file,
                    document_name=r.document_name,
                    page_number=r.page_number,
                    chunk_id=r.chunk.id,
                    text_snippet=first_line,
                )
            )

        return answer, active_provider, citations
