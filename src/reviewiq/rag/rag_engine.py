"""Grounded Retrieval-Augmented Generation (RAG) Engine using Google Gemini.

Answers user questions about products strictly grounded in retrieved reviews:
- Uses HybridRetriever to fetch top-K relevant review evidence
- Builds structured prompts with review citations and anti-hallucination guardrails
- Interacts with Google Gemini (e.g., gemini-1.5-flash) via google.generativeai
- Gracefully handles missing API keys, rate limits, and offline/CI environments
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from reviewiq.retrieval.hybrid_search import HybridRetriever, SearchResult
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewCitation:
    """Evidence citation sourced from a customer review."""

    citation_id: int
    doc_id: str | int
    text: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "relevance_score": round(self.relevance_score, 4),
            "metadata": self.metadata,
        }


@dataclass
class RAGResponse:
    """Grounded answer and citations returned by the RAG engine."""

    answer: str
    query: str
    citations: list[ReviewCitation]
    model_name: str
    is_grounded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "query": self.query,
            "citations": [c.to_dict() for c in self.citations],
            "model_name": self.model_name,
            "is_grounded": self.is_grounded,
            "metadata": self.metadata,
        }


class ReviewRAGEngine:
    """Retrieval-Augmented Generation engine grounded in product customer reviews.

    Parameters
    ----------
    retriever:
        Optional indexed HybridRetriever instance.
    api_key:
        Google Gemini API key. Defaults to GEMINI_API_KEY environment variable.
    model_name:
        Gemini model identifier (e.g. 'gemini-1.5-flash').
    top_k:
        Number of context reviews to retrieve. Default from config (8).
    temperature:
        Generation temperature. Default 0.1 for deterministic, factual output.
    """

    SYSTEM_INSTRUCTION = (
        "You are ReviewIQ Assistant, an expert AI product advisor grounded strictly "
        "in authentic customer reviews.\n"
        "Guidelines:\n"
        "1. Answer the question using ONLY the provided customer review excerpts below.\n"
        "2. Cite your sources using citation markers like [Citation 1], [Citation 2] "
        "when stating facts or customer experiences.\n"
        "3. If the provided reviews do NOT contain sufficient information to answer the "
        "question, explicitly say: 'Based on the available customer reviews, there is "
        "not enough information to answer this question.'\n"
        "4. Never extrapolate, hallucinate, or use outside product specifications not in the text."
    )

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        cfg = load_config()
        rag_cfg = getattr(cfg, "rag", None)

        self.retriever = retriever
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = (
            model_name
            if model_name is not None
            else getattr(rag_cfg, "gemini_model", "gemini-1.5-flash")
        )
        self.top_k = top_k if top_k is not None else getattr(rag_cfg, "retrieval_k", 8)
        self.temperature = (
            temperature if temperature is not None else getattr(rag_cfg, "temperature", 0.1)
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else getattr(rag_cfg, "max_output_tokens", 512)
        )

        self._gemini_client = None

    def _get_gemini_model(self) -> Any:
        """Lazily initialize Google Generative AI client."""
        if not self.api_key:
            return None

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_INSTRUCTION,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            return model
        except Exception as exc:
            logger.warning("Could not initialize Gemini model '%s': %s", self.model_name, exc)
            return None

    def build_prompt(self, query: str, search_results: Sequence[SearchResult]) -> str:
        """Construct the grounded context prompt from retrieved review excerpts.

        Parameters
        ----------
        query:
            Natural language user inquiry.
        search_results:
            List of SearchResult objects retrieved by hybrid search.

        Returns
        -------
        str:
            Formatted prompt text.
        """
        if not search_results:
            return f"Question: {query}\n\nNo relevant customer reviews found."

        context_lines: list[str] = ["Here are relevant excerpts from verified customer reviews:\n"]
        for i, res in enumerate(search_results, start=1):
            product_tag = f" (Product: {res.doc_id})" if res.doc_id else ""
            context_lines.append(f"[Citation {i}]{product_tag}\n\"{res.text.strip()}\"\n")

        context_lines.append(
            f"User Question: {query}\n\n"
            "Please answer the question based strictly on the reviews above, citing [Citation X] for each fact:"
        )
        return "\n".join(context_lines)

    def answer_query(
        self,
        query: str,
        retrieved_reviews: Sequence[SearchResult] | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Answer a product question using retrieved review context.

        Parameters
        ----------
        query:
            User question.
        retrieved_reviews:
            Optional pre-retrieved SearchResult list. If None, queries self.retriever.
        top_k:
            Number of reviews to retrieve if self.retriever is called.

        Returns
        -------
        RAGResponse:
            Structured answer with citations and metadata.
        """
        k = top_k or self.top_k

        # 1. Retrieve evidence reviews
        if retrieved_reviews is not None:
            results = list(retrieved_reviews)[:k]
        elif self.retriever is not None and self.retriever.is_indexed:
            results = self.retriever.search(query, candidate_k=k * 2, final_k=k, rerank=True)
        else:
            results = []

        citations: list[ReviewCitation] = [
            ReviewCitation(
                citation_id=i,
                doc_id=res.doc_id,
                text=res.text,
                relevance_score=res.score,
                metadata=res.metadata,
            )
            for i, res in enumerate(results, start=1)
        ]

        if not citations:
            return RAGResponse(
                answer="No relevant customer reviews were found to answer your inquiry.",
                query=query,
                citations=[],
                model_name=self.model_name,
                is_grounded=False,
            )

        # 2. Build grounded prompt
        prompt = self.build_prompt(query, results)

        # 3. Call Gemini
        model = self._get_gemini_model()
        if model is not None:
            try:
                response = model.generate_content(prompt)
                answer_text = response.text.strip()
                return RAGResponse(
                    answer=answer_text,
                    query=query,
                    citations=citations,
                    model_name=self.model_name,
                    is_grounded=True,
                )
            except Exception as exc:
                logger.error("Gemini API call failed: %s; falling back to extractive summary", exc)

        # 4. Fallback when API key is unset or network/quota error occurs
        fallback_points = []
        for c in citations[:3]:
            short_excerpt = c.text[:120].strip()
            fallback_points.append(f"- [Citation {c.citation_id}]: \"{short_excerpt}...\"")

        fallback_answer = (
            f"Based on {len(citations)} retrieved customer reviews, here are relevant excerpts:\n"
            + "\n".join(fallback_points)
            + "\n\n*(Note: Set GEMINI_API_KEY in your environment for full generative AI synthesis.)*"
        )

        return RAGResponse(
            answer=fallback_answer,
            query=query,
            citations=citations,
            model_name=f"{self.model_name} (offline/fallback)",
            is_grounded=True,
        )
