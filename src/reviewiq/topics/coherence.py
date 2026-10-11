"""Topic coherence and diversity evaluation suite for ReviewIQ.

Implements intrinsic topic quality evaluation metrics:
- UMass Coherence (C_UMass):
    C_{UMass}(w_1...w_M) = [2 / (M*(M-1))] * sum_{m=2}^M sum_{l=1}^{m-1} log[(D(w_m, w_l) + eps) / D(w_l)]
- Normalized Pointwise Mutual Information (NPMI):
    NPMI(w_m, w_l) = log[P(w_m, w_l) / (P(w_m)*P(w_l))] / -log[P(w_m, w_l)]
- Topic Diversity:
    Fraction of unique words across top keywords of all topics (higher = less redundant)
- Multi-model benchmarking: Compares LDA, NMF, and BERTopic in an automated evaluation table
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_document_frequencies(
    corpus_tokens: Sequence[Sequence[str]],
) -> tuple[int, dict[str, int], dict[tuple[str, str], int]]:
    """Compute document frequencies D(w) and co-document frequencies D(w1, w2).

    Parameters
    ----------
    corpus_tokens:
        List of tokenized documents (each document is a sequence of words).

    Returns
    -------
    (total_docs, doc_freqs, co_doc_freqs)
    """
    total_docs = len(corpus_tokens)
    doc_freqs: dict[str, int] = Counter()
    co_doc_freqs: dict[tuple[str, str], int] = Counter()

    for doc in corpus_tokens:
        unique_tokens = set(doc)
        for token in unique_tokens:
            doc_freqs[token] += 1

        # Sort pairwise to ensure deterministic key order
        sorted_tokens = sorted(unique_tokens)
        for i in range(len(sorted_tokens)):
            w1 = sorted_tokens[i]
            for j in range(i + 1, len(sorted_tokens)):
                w2 = sorted_tokens[j]
                co_doc_freqs[(w1, w2)] += 1

    return total_docs, doc_freqs, co_doc_freqs


def calculate_umass_coherence(
    topic_words: Sequence[str],
    corpus_tokens: Sequence[Sequence[str]],
    top_n: int = 10,
    eps: float = 1e-12,
) -> float:
    """Calculate UMass topic coherence for a single list of topic words.

    Parameters
    ----------
    topic_words:
        Top keywords of the topic ordered by weight descending.
    corpus_tokens:
        Tokenized corpus documents.
    top_n:
        Number of top words to evaluate.
    eps:
        Smoothing parameter to avoid log(0).

    Returns
    -------
    Mean UMass coherence score (typically negative; closer to 0 is better).
    """
    words = list(topic_words)[:top_n]
    M = len(words)
    if M < 2:
        return 0.0

    _, doc_freqs, co_doc_freqs = _compute_document_frequencies(corpus_tokens)

    scores: list[float] = []
    for m in range(1, M):
        w_m = words[m]
        for l in range(0, m):
            w_l = words[l]

            # Lookup co-document frequency
            pair = (min(w_m, w_l), max(w_m, w_l))
            d_ml = co_doc_freqs.get(pair, 0)
            d_l = doc_freqs.get(w_l, 0)

            score = np.log((d_ml + eps) / (d_l + eps))
            scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


def calculate_npmi_coherence(
    topic_words: Sequence[str],
    corpus_tokens: Sequence[Sequence[str]],
    top_n: int = 10,
    eps: float = 1e-12,
) -> float:
    """Calculate Normalized Pointwise Mutual Information (NPMI) coherence.

    Normalized to [-1, 1], where 1 represents perfect co-occurrence.
    """
    words = list(topic_words)[:top_n]
    M = len(words)
    if M < 2:
        return 0.0

    N, doc_freqs, co_doc_freqs = _compute_document_frequencies(corpus_tokens)
    if N == 0:
        return 0.0

    scores: list[float] = []
    for i in range(M):
        w_i = words[i]
        p_i = (doc_freqs.get(w_i, 0) + eps) / N

        for j in range(i + 1, M):
            w_j = words[j]
            p_j = (doc_freqs.get(w_j, 0) + eps) / N

            pair = (min(w_i, w_j), max(w_i, w_j))
            co_count = co_doc_freqs.get(pair, 0)
            if co_count == 0:
                scores.append(-1.0)
                continue

            p_ij = (co_count + eps) / N

            # PMI = log(p_ij / (p_i * p_j))
            pmi = np.log(p_ij / (p_i * p_j))
            # NPMI = PMI / (-log(p_ij))
            npmi = pmi / (-np.log(p_ij))
            scores.append(float(npmi))

    return float(np.mean(scores)) if scores else 0.0


def calculate_topic_diversity(
    topics: dict[int, list[tuple[str, float]]] | list[list[str]],
    top_n: int = 10,
) -> float:
    """Calculate topic diversity (fraction of unique words across topics).

    Range [0, 1]. A score of 1.0 means complete vocabulary diversity across topics.
    """
    all_words: list[str] = []

    if isinstance(topics, dict):
        for word_list in topics.values():
            all_words.extend([w[0] if isinstance(w, tuple) else str(w) for w in word_list[:top_n]])
    else:
        for word_list in topics:
            all_words.extend([str(w) for w in word_list[:top_n]])

    if not all_words:
        return 0.0

    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


def evaluate_topic_model(
    topics: dict[int, list[tuple[str, float]]] | list[list[str]],
    corpus_tokens: Sequence[Sequence[str]],
    top_n: int = 10,
) -> dict[str, float]:
    """Calculate mean UMass, NPMI, and diversity for all topics in a model.

    Returns
    -------
    dict with keys: 'umass_coherence', 'npmi_coherence', 'topic_diversity'
    """
    # Extract plain word lists
    topic_word_lists: list[list[str]] = []
    if isinstance(topics, dict):
        for words in topics.values():
            topic_word_lists.append([w[0] if isinstance(w, tuple) else str(w) for w in words[:top_n]])
    else:
        for words in topics:
            topic_word_lists.append([str(w) for w in words[:top_n]])

    umass_scores = [
        calculate_umass_coherence(w_list, corpus_tokens, top_n=top_n)
        for w_list in topic_word_lists
    ]
    npmi_scores = [
        calculate_npmi_coherence(w_list, corpus_tokens, top_n=top_n)
        for w_list in topic_word_lists
    ]
    diversity = calculate_topic_diversity(topics, top_n=top_n)

    return {
        "umass_coherence": float(np.mean(umass_scores)) if umass_scores else 0.0,
        "npmi_coherence": float(np.mean(npmi_scores)) if npmi_scores else 0.0,
        "topic_diversity": float(diversity),
    }


def compare_topic_models(
    models: dict[str, Any],
    corpus_tokens: Sequence[Sequence[str]],
    top_n: int = 10,
) -> pd.DataFrame:
    """Benchmark and compare multiple topic models (LDA, NMF, BERTopic).

    Parameters
    ----------
    models:
        Dict mapping model name -> topic model instance (must implement .get_topics()).
    corpus_tokens:
        Tokenized reference corpus.
    top_n:
        Number of top keywords to evaluate per topic.

    Returns
    -------
    pandas DataFrame summarizing metrics for each model.
    """
    rows = []
    for name, model in models.items():
        if hasattr(model, "get_topics"):
            topics = model.get_topics(top_n=top_n)
        else:
            raise ValueError(f"Model '{name}' must implement get_topics(top_n).")

        metrics = evaluate_topic_model(topics, corpus_tokens, top_n=top_n)
        sample_topic = topics.get(0, [])
        sample_words = ", ".join([w[0] for w in sample_topic[:5]]) if sample_topic else "N/A"

        rows.append({
            "Model": name,
            "Topics": len(topics),
            "UMass Coherence": f"{metrics['umass_coherence']:.4f}",
            "NPMI Coherence": f"{metrics['npmi_coherence']:.4f}",
            "Topic Diversity": f"{metrics['topic_diversity']:.4f}",
            "Sample Topic Keywords": sample_words,
        })

    df = pd.DataFrame(rows)
    return df
