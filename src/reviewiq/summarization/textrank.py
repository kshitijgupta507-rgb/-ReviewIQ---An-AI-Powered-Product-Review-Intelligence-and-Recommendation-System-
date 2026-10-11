"""Extractive text summarization using the TextRank graph algorithm.

Implements the classical TextRank algorithm (Mihalcea & Tarau, 2004):
- Segments multi-document or single review text into individual sentences
- Builds a sentence-sentence similarity graph using TF-IDF cosine similarity
- Runs Google PageRank on the graph to rank sentence importance
- Selects top-ranked sentences and orders them in original chronological sequence
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class TextRankSummarizer:
    """Extractive summarizer based on graph PageRank over sentence similarities.

    Parameters
    ----------
    num_sentences:
        Number of sentences to extract for the summary. Default from config (5).
    similarity_threshold:
        Minimum cosine similarity required to create an edge between sentences.
        Default from config (0.1).
    damping_factor:
        PageRank damping factor (alpha). Default 0.85.
    max_iter:
        Maximum PageRank power-iteration iterations. Default 200.
    """

    def __init__(
        self,
        num_sentences: int | None = None,
        similarity_threshold: float | None = None,
        damping_factor: float = 0.85,
        max_iter: int = 200,
    ) -> None:
        cfg = load_config().get("summarization", {}).get("textrank", {})
        self.num_sentences = num_sentences or cfg.get("num_sentences", 5)
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else cfg.get("similarity_threshold", 0.1)
        )
        self.damping_factor = damping_factor
        self.max_iter = max_iter

    def split_sentences(self, text: str) -> list[str]:
        """Split text into clean, non-empty sentences.

        Parameters
        ----------
        text:
            Input text document.

        Returns
        -------
        list[str]:
            List of segmented sentences.
        """
        if not text or not text.strip():
            return []

        # Split on sentence boundaries (. ! ?) followed by whitespace or line breaks
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        sentences: list[str] = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Retain sentences with sufficient content (> 10 characters or >= 3 words)
            if len(s_clean) >= 10 and len(s_clean.split()) >= 3:
                sentences.append(s_clean)
            elif len(s_clean) > 0 and len(raw_sentences) < 3:
                # Include shorter sentence if very few input sentences exist
                sentences.append(s_clean)

        return sentences

    def summarize(
        self,
        text_or_reviews: str | Sequence[str],
        num_sentences: int | None = None,
    ) -> str:
        """Generate an extractive summary of text or a collection of reviews.

        Parameters
        ----------
        text_or_reviews:
            Single review text string, or sequence of review strings.
        num_sentences:
            Number of sentences to return. Defaults to instance configuration.

        Returns
        -------
        str:
            Concatenated extractive summary sentences.
        """
        result = self.summarize_with_scores(text_or_reviews, num_sentences=num_sentences)
        return result["summary"]

    def summarize_with_scores(
        self,
        text_or_reviews: str | Sequence[str],
        num_sentences: int | None = None,
    ) -> dict[str, Any]:
        """Generate an extractive summary along with sentence PageRank scores.

        Parameters
        ----------
        text_or_reviews:
            Single review text string, or sequence of review strings.
        num_sentences:
            Number of sentences to extract.

        Returns
        -------
        dict[str, Any]:
            Dictionary containing 'summary', 'ranked_sentences', 'scores',
            'num_input_sentences', and 'graph_edges'.
        """
        k = num_sentences if num_sentences is not None else self.num_sentences

        # 1. Unpack and segment sentences
        if isinstance(text_or_reviews, str):
            sentences = self.split_sentences(text_or_reviews)
        else:
            sentences = []
            for item in text_or_reviews:
                if isinstance(item, str) and item.strip():
                    sentences.extend(self.split_sentences(item))

        # Deduplicate identical adjacent sentences while preserving order
        unique_sentences: list[str] = []
        seen = set()
        for s in sentences:
            s_lower = s.lower().strip()
            if s_lower not in seen:
                seen.add(s_lower)
                unique_sentences.append(s)
        sentences = unique_sentences

        n_sentences = len(sentences)
        if n_sentences == 0:
            return {
                "summary": "",
                "ranked_sentences": [],
                "scores": {},
                "num_input_sentences": 0,
                "graph_edges": 0,
            }

        # If sentences <= requested count, return all preserving order
        if n_sentences <= k:
            return {
                "summary": " ".join(sentences),
                "ranked_sentences": [(i, sentences[i], 1.0 / n_sentences) for i in range(n_sentences)],
                "scores": {i: 1.0 / n_sentences for i in range(n_sentences)},
                "num_input_sentences": n_sentences,
                "graph_edges": 0,
            }

        # 2. Vectorize sentences using TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=1, token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        except Exception as exc:
            logger.warning("TF-IDF vectorization failed for sentences: %s; falling back to lead sentences", exc)
            lead_summary = " ".join(sentences[:k])
            return {
                "summary": lead_summary,
                "ranked_sentences": [(i, sentences[i], 1.0) for i in range(min(k, n_sentences))],
                "scores": {i: 1.0 for i in range(min(k, n_sentences))},
                "num_input_sentences": n_sentences,
                "graph_edges": 0,
            }

        # 3. Construct NetworkX sentence similarity graph
        graph = nx.Graph()
        for i in range(n_sentences):
            graph.add_node(i)

        edge_count = 0
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                weight = float(sim_matrix[i, j])
                if weight >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=weight)
                    edge_count += 1

        # 4. Compute PageRank
        try:
            if edge_count > 0:
                scores = nx.pagerank(
                    graph,
                    alpha=self.damping_factor,
                    max_iter=self.max_iter,
                    weight="weight",
                )
            else:
                # Disconnected graph - uniform PageRank
                scores = {i: 1.0 / n_sentences for i in range(n_sentences)}
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank power iteration failed to converge; using uniform weights")
            scores = {i: 1.0 / n_sentences for i in range(n_sentences)}

        # 5. Rank sentences by PageRank score
        ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        top_indices = ranked_indices[:k]

        # 6. Sort selected indices chronologically to preserve narrative coherence
        top_indices_chronological = sorted(top_indices)
        selected_sentences = [sentences[idx] for idx in top_indices_chronological]
        summary_text = " ".join(selected_sentences)

        ranked_details = [(idx, sentences[idx], float(scores[idx])) for idx in ranked_indices]

        return {
            "summary": summary_text,
            "ranked_sentences": ranked_details,
            "scores": {idx: float(score) for idx, score in scores.items()},
            "num_input_sentences": n_sentences,
            "graph_edges": edge_count,
        }
