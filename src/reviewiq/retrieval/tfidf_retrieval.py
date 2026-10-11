"""TF-IDF lexical retrieval module for ReviewIQ.

Implements classical sparse Information Retrieval (IR):
- Builds inverted index / TF-IDF sparse matrix over review corpus
- Computes cosine similarity between query and indexed documents
- Fast sparse dot-product scoring (sublinear TF scaling)
- Returns structured SearchResult objects with score, rank, and metadata
- Model artifact persistence via joblib
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.metrics.pairwise import cosine_similarity

from reviewiq.representations.tfidf import TfidfVectorizerWrapper
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Standardized retrieval result representation across all search engines."""

    doc_id: str | int
    text: str
    score: float
    rank: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": float(self.score),
            "rank": int(self.rank),
            "metadata": self.metadata,
        }


class TFIDFRetriever:
    """Sparse lexical search engine using TF-IDF representation and cosine similarity.

    Parameters
    ----------
    max_features:
        Vocabulary size cap. Default from config (50000).
    ngram_range:
        N-gram tuple (min_n, max_n). Default (1, 2).
    sublinear_tf:
        Whether to apply sublinear TF scaling (1 + log(tf)). Default True.
    """

    def __init__(
        self,
        max_features: int | None = None,
        min_df: int | float | None = None,
        max_df: float | None = None,
        ngram_range: tuple[int, int] | None = None,
        sublinear_tf: bool | None = None,
    ) -> None:
        cfg = load_config()
        r_cfg = getattr(cfg, "retrieval", None)
        rep_cfg = getattr(getattr(cfg, "representations", None), "tfidf", None)

        self.max_features = (
            max_features
            if max_features is not None
            else getattr(rep_cfg, "max_features", 50000)
        )
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = (
            ngram_range
            if ngram_range is not None
            else tuple(getattr(rep_cfg, "ngram_range", [1, 2]))
        )
        self.sublinear_tf = (
            sublinear_tf
            if sublinear_tf is not None
            else getattr(rep_cfg, "sublinear_tf", True)
        )
        self.default_top_k = getattr(r_cfg, "top_k", 20) if r_cfg else 20

        self.vectorizer = TfidfVectorizerWrapper(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
        )

        self.doc_matrix: spmatrix | None = None
        self.documents: list[str] = []
        self.doc_ids: list[str | int] = []
        self.metadata_list: list[dict[str, Any]] = []
        self.is_indexed = False

    def index(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str | int] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> "TFIDFRetriever":
        """Index a corpus of text documents.

        Parameters
        ----------
        documents:
            List of raw or cleaned review texts.
        doc_ids:
            Optional unique identifier per document. If None, uses integer indices 0..N-1.
        metadata:
            Optional metadata dictionaries corresponding to each document.

        Returns
        -------
        self
        """
        n_docs = len(documents)
        if n_docs == 0:
            raise ValueError("Cannot index an empty document collection.")

        logger.info(f"Indexing {n_docs} documents with TF-IDF...")
        self.documents = list(documents)
        self.doc_ids = (
            list(doc_ids) if doc_ids is not None else [str(i) for i in range(n_docs)]
        )
        self.metadata_list = (
            list(metadata) if metadata is not None else [{} for _ in range(n_docs)]
        )

        # Adapt min_df / max_df dynamically if dataset size is small
        actual_min = self.min_df
        actual_max = self.max_df
        if n_docs < 10:
            actual_min = 1
            actual_max = 1.0
        elif actual_min is not None and isinstance(actual_min, int) and actual_min > n_docs:
            actual_min = 1

        self.vectorizer = TfidfVectorizerWrapper(
            max_features=self.max_features,
            min_df=actual_min,
            max_df=actual_max,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
        )

        self.doc_matrix = self.vectorizer.fit_transform(self.documents)
        self.is_indexed = True

        logger.info(
            f"TF-IDF indexing complete. Matrix shape: {self.doc_matrix.shape}, "
            f"vocab: {len(self.vectorizer.vocabulary)}"
        )
        return self

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Retrieve top matching documents for query string.

        Parameters
        ----------
        query:
            Search query.
        top_k:
            Number of top results to return. If None, uses default_top_k.
        min_score:
            Score threshold below which results are excluded.

        Returns
        -------
        Ranked list of SearchResult objects sorted descending by cosine similarity.
        """
        if not self.is_indexed or self.doc_matrix is None:
            raise RuntimeError("TFIDFRetriever must be indexed before querying.")

        k = top_k if top_k is not None else self.default_top_k
        if not query or not query.strip():
            return []

        # Transform query vector
        query_vec = self.vectorizer.transform([query])

        # If query has zero terms in vocabulary, cosine similarity is 0
        if query_vec.nnz == 0:
            return []

        # Normalized TF-IDF matrix dot product produces exact cosine similarity
        scores = self.doc_matrix.dot(query_vec.T).toarray().ravel()

        # Find top k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[SearchResult] = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score or score <= 0.0:
                continue

            results.append(
                SearchResult(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=score,
                    rank=rank,
                    metadata=self.metadata_list[idx],
                )
            )
            rank += 1

        return results

    def save(self, filepath: str | Path) -> None:
        """Serialize index and documents to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "doc_matrix": self.doc_matrix,
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadata_list": self.metadata_list,
                "is_indexed": self.is_indexed,
            },
            path,
        )
        logger.info(f"Saved TFIDFRetriever index to {path}")

    @classmethod
    def load(cls, filepath: str | Path) -> "TFIDFRetriever":
        """Load serialized TFIDFRetriever index from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"TFIDFRetriever file not found at {path}")
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.doc_matrix = data["doc_matrix"]
        instance.documents = data["documents"]
        instance.doc_ids = data["doc_ids"]
        instance.metadata_list = data["metadata_list"]
        instance.is_indexed = data["is_indexed"]
        logger.info(f"Loaded TFIDFRetriever index from {path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews

    try:
        df = load_reviews()
        texts = df["review_clean"].tolist() if "review_clean" in df.columns else df["text"].tolist()
        asins = df["parent_asin"].tolist() if "parent_asin" in df.columns else None
        sample_texts = texts[:500]
        sample_asins = asins[:500] if asins else None
    except FileNotFoundError:
        print("Real data not yet downloaded. Running demonstration on sample review corpus.")
        sample_texts = [
            "Battery life is amazing and lasts 12 hours without charging. Highly recommend!",
            "The battery drains super fast and dies in two hours of gaming or browsing.",
            "Incredible battery performance, great standby time and fast USB-C charging.",
            "Display screen is crisp colorful vibrant OLED with great brightness and HDR.",
            "The screen is dim washed out with terrible viewing angles and glare.",
            "Solid aluminum chassis, premium metal build quality and sturdy smooth hinges.",
            "Cheap fragile plastic build quality, creaky body and loose wobbly hinges.",
            "Fast performance, smooth multitasking with 16GB RAM and speedy SSD.",
            "Lags constantly, thermal throttles under load with loud annoying fans.",
            "Great value for money, affordable price for high-end laptop specs.",
        ] * 5
        sample_asins = [f"ASIN_{i}" for i in range(len(sample_texts))]

    retriever = TFIDFRetriever(max_features=1000)
    retriever.index(sample_texts, doc_ids=sample_asins)

    query = "battery charging duration"
    print(f"\n--- Top Results for Query: '{query}' ---")
    results = retriever.search(query, top_k=3)
    for r in results:
        print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.doc_id} | Text: {r.text[:80]}...")
