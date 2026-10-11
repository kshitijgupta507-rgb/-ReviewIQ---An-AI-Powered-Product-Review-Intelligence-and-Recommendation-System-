"""Latent Dirichlet Allocation (LDA) topic modeling for ReviewIQ.

Implements generative probabilistic topic modeling via Latent Dirichlet Allocation:
- Discovers latent thematic topics in review corpora from Bag-of-Words count matrices
- Infers document-topic distributions and topic-word distributions
- Computes topic perplexity and approximate log-likelihood
- Extracts top keywords with probabilities for each latent topic
- Model persistence via joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from sklearn.decomposition import LatentDirichletAllocation

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class LDATopicModel:
    """Latent Dirichlet Allocation (LDA) Topic Model wrapper.

    Parameters
    ----------
    n_topics:
        Number of topics to extract. Default from config `topics.num_topics` (10).
    max_iter:
        Maximum iterations for estimation. Default from config `topics.lda.iterations` (400).
    learning_method:
        Learning method: 'batch' or 'online'. Default 'batch'.
    alpha:
        Prior on document-topic distribution (doc_topic_prior).
        If 'auto' or None, scikit-learn uses 1.0 / n_topics.
    eta:
        Prior on topic-word distribution (topic_word_prior).
        If 'auto' or None, scikit-learn uses 1.0 / n_topics.
    random_state:
        Seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_topics: int | None = None,
        max_iter: int | None = None,
        learning_method: str = "batch",
        alpha: float | str | None = None,
        eta: float | str | None = None,
        random_state: int = 42,
    ) -> None:
        cfg = load_config()
        t_cfg = getattr(cfg, "topics", None)
        lda_cfg = getattr(t_cfg, "lda", None) if t_cfg else None

        self.n_topics = n_topics if n_topics is not None else getattr(t_cfg, "num_topics", 10)
        self.max_iter = max_iter if max_iter is not None else getattr(lda_cfg, "iterations", 50)
        self.learning_method = learning_method
        self.random_state = random_state

        # Resolve doc_topic_prior (alpha)
        resolved_alpha = alpha if alpha is not None else getattr(lda_cfg, "alpha", "auto")
        if isinstance(resolved_alpha, str) and resolved_alpha.lower() == "auto":
            self.doc_topic_prior = None
        elif isinstance(resolved_alpha, (int, float)):
            self.doc_topic_prior = float(resolved_alpha)
        else:
            self.doc_topic_prior = None

        # Resolve topic_word_prior (eta)
        resolved_eta = eta if eta is not None else getattr(lda_cfg, "eta", "auto")
        if isinstance(resolved_eta, str) and resolved_eta.lower() == "auto":
            self.topic_word_prior = None
        elif isinstance(resolved_eta, (int, float)):
            self.topic_word_prior = float(resolved_eta)
        else:
            self.topic_word_prior = None

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "LDATopicModel":
        """Fit LDA model on term count matrix X (typically BoW).

        Parameters
        ----------
        X:
            Document-term count matrix (shape: n_documents, n_features).
        feature_names:
            Vocabulary feature names corresponding to columns of X.

        Returns
        -------
        self
        """
        logger.info(
            f"Fitting LatentDirichletAllocation with {self.n_topics} topics, max_iter={self.max_iter}..."
        )
        self.model.fit(X)
        self.is_fitted = True

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names = list(getattr(X, "columns"))
        else:
            self.feature_names = [f"word_{i}" for i in range(X.shape[1])]

        logger.info("LDA fitting complete.")
        return self

    def transform(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Compute document-topic distributions for X.

        Parameters
        ----------
        X:
            Document-term matrix.

        Returns
        -------
        Document-topic probability matrix of shape (n_samples, n_topics).
        """
        if not self.is_fitted:
            raise RuntimeError("LDATopicModel must be fitted before calling transform().")
        return self.model.transform(X)

    def fit_transform(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Fit LDA model and return document-topic distributions.

        Parameters
        ----------
        X:
            Document-term count matrix.
        feature_names:
            Vocabulary feature names.

        Returns
        -------
        Document-topic probabilities of shape (n_samples, n_topics).
        """
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def get_topic_words(self, topic_idx: int, top_n: int = 10) -> list[tuple[str, float]]:
        """Retrieve top keywords and their normalized probabilities for a topic.

        Parameters
        ----------
        topic_idx:
            Index of the topic (0 to n_topics - 1).
        top_n:
            Number of top words to return.

        Returns
        -------
        List of (word, probability) tuples sorted descending by probability.
        """
        if not self.is_fitted:
            raise RuntimeError("LDATopicModel must be fitted before extracting topic words.")
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise ValueError(f"topic_idx {topic_idx} out of range [0, {self.n_topics - 1}].")

        # components_ has shape (n_components, n_features) with pseudocounts
        topic_dist = self.model.components_[topic_idx]
        total_weight = topic_dist.sum()
        normalized_dist = topic_dist / (total_weight + 1e-12)

        top_indices = np.argsort(normalized_dist)[::-1][:top_n]
        return [
            (
                self.feature_names[idx] if idx < len(self.feature_names) else f"word_{idx}",
                float(normalized_dist[idx]),
            )
            for idx in top_indices
        ]

    def get_topics(self, top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        """Extract top keywords for all discovered topics.

        Parameters
        ----------
        top_n:
            Number of words per topic.

        Returns
        -------
        Dictionary mapping topic_id -> list of (word, probability).
        """
        return {k: self.get_topic_words(k, top_n=top_n) for k in range(self.n_topics)}

    def get_document_topics(
        self, X: spmatrix | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get dominant topic, confidence score, and full distribution for each document.

        Parameters
        ----------
        X:
            Document-term matrix.

        Returns
        -------
        (dominant_topics, topic_confidences, topic_distributions)
        """
        doc_topics = self.transform(X)
        dominant_topics = np.argmax(doc_topics, axis=1)
        topic_confidences = np.max(doc_topics, axis=1)
        return dominant_topics, topic_confidences, doc_topics

    def perplexity(self, X: spmatrix | np.ndarray) -> float:
        """Calculate model perplexity on document-term matrix X.

        Lower perplexity indicates better predictive generalization.
        """
        if not self.is_fitted:
            raise RuntimeError("LDATopicModel must be fitted before computing perplexity.")
        return float(self.model.perplexity(X))

    def score(self, X: spmatrix | np.ndarray) -> float:
        """Calculate approximate log-likelihood on document-term matrix X.

        Higher score indicates better fit.
        """
        if not self.is_fitted:
            raise RuntimeError("LDATopicModel must be fitted before computing score.")
        return float(self.model.score(X))

    def save(self, path: str | Path) -> None:
        """Serialize fitted model to disk using joblib."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_topics": self.n_topics,
                "max_iter": self.max_iter,
                "learning_method": self.learning_method,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
            },
            save_path,
        )
        logger.info(f"Saved LDATopicModel to {save_path}")

    @classmethod
    def load(cls, path: str | Path) -> "LDATopicModel":
        """Load serialized LDATopicModel from disk."""
        data = joblib.load(path)
        instance = cls(
            n_topics=data["n_topics"],
            max_iter=data["max_iter"],
            learning_method=data.get("learning_method", "batch"),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data["is_fitted"]
        logger.info(f"Loaded LDATopicModel from {path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews
    from reviewiq.representations.bow import BagOfWordsVectorizer

    try:
        df = load_reviews()
        texts = df["review_clean"].tolist() if "review_clean" in df.columns else df["text"].tolist()
    except FileNotFoundError:
        print("Real data not yet downloaded. Running demonstration on sample review corpus.")
        texts = [
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
        ] * 10

    print(f"Vectorizing {len(texts)} reviews with Bag-of-Words...")
    bow = BagOfWordsVectorizer(max_features=1000)
    X = bow.fit_transform(texts)

    lda = LDATopicModel(n_topics=4, max_iter=20)
    lda.fit(X, feature_names=bow.feature_names_)

    print("\n--- Discovered LDA Topics ---")
    for t_id, words in lda.get_topics(top_n=5).items():
        words_str = ", ".join([f"{w} ({p:.3f})" for w, p in words])
        print(f"Topic {t_id}: {words_str}")


