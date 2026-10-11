"""Non-negative Matrix Factorization (NMF) topic modeling for ReviewIQ.

Implements linear-algebraic non-negative matrix factorization:
- Decomposes non-negative document-term matrix V ≈ W x H into:
  - W (Document-Topic matrix): mixture weights of topics for each review
  - H (Topic-Word matrix): importance weights of vocabulary words for each topic
- Optimizes Frobenius or Kullback-Leibler divergence with NNDSVD initialization
- Computes reconstruction error ||V - WH||
- Extracts distinctive keywords per topic with normalized weights
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
from sklearn.decomposition import NMF

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class NMFTopicModel:
    """Non-negative Matrix Factorization (NMF) Topic Model wrapper.

    Parameters
    ----------
    n_topics:
        Number of topics to factorize. Default from config `topics.num_topics` (10).
    init:
        Initialization method ('nndsvda', 'nndsvd', 'random'). Default from config `topics.nmf.init` ('nndsvda').
    max_iter:
        Maximum iterations. Default from config `topics.nmf.max_iter` (500).
    beta_loss:
        Loss function to optimize ('frobenius' or 'kullback-leibler'). Default 'frobenius'.
    solver:
        Numerical solver ('cd' for coordinate descent, 'mu' for multiplicative update). Default 'cd'.
    random_state:
        Seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_topics: int | None = None,
        init: str | None = None,
        max_iter: int | None = None,
        beta_loss: str = "frobenius",
        solver: str = "cd",
        random_state: int = 42,
    ) -> None:
        cfg = load_config()
        t_cfg = getattr(cfg, "topics", None)
        nmf_cfg = getattr(t_cfg, "nmf", None) if t_cfg else None

        self.n_topics = n_topics if n_topics is not None else getattr(t_cfg, "num_topics", 10)
        self.init = init if init is not None else getattr(nmf_cfg, "init", "nndsvda")
        self.max_iter = max_iter if max_iter is not None else getattr(nmf_cfg, "max_iter", 500)
        self.beta_loss = beta_loss
        self.solver = solver
        self.random_state = random_state

        self.model = NMF(
            n_components=self.n_topics,
            init=self.init,
            max_iter=self.max_iter,
            beta_loss=self.beta_loss,
            solver=self.solver,
            random_state=self.random_state,
        )

        self.feature_names: list[str] = []
        self.reconstruction_err_: float | None = None
        self.is_fitted = False

    def fit(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "NMFTopicModel":
        """Fit NMF model on non-negative feature matrix X (typically TF-IDF).

        Parameters
        ----------
        X:
            Non-negative document-term matrix (shape: n_documents, n_features).
        feature_names:
            Vocabulary feature names corresponding to columns of X.

        Returns
        -------
        self
        """
        logger.info(
            f"Fitting NMF with {self.n_topics} components, init='{self.init}', max_iter={self.max_iter}..."
        )
        self.model.fit(X)
        self.reconstruction_err_ = float(self.model.reconstruction_err_)
        self.is_fitted = True

        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names = list(getattr(X, "columns"))
        else:
            self.feature_names = [f"word_{i}" for i in range(X.shape[1])]

        logger.info(
            f"NMF fitting complete. Final reconstruction error: {self.reconstruction_err_:.4f}"
        )
        return self

    def transform(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Compute document-topic representation matrix W.

        Parameters
        ----------
        X:
            Non-negative document-term matrix.

        Returns
        -------
        Document-topic matrix W of shape (n_samples, n_topics).
        """
        if not self.is_fitted:
            raise RuntimeError("NMFTopicModel must be fitted before calling transform().")
        return self.model.transform(X)

    def fit_transform(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Fit NMF model and return document-topic representation matrix W.

        Parameters
        ----------
        X:
            Non-negative document-term matrix.
        feature_names:
            Vocabulary feature names.

        Returns
        -------
        Document-topic matrix W of shape (n_samples, n_topics).
        """
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    @property
    def components_(self) -> np.ndarray:
        """Topic-word matrix H of shape (n_topics, n_features)."""
        if not self.is_fitted:
            raise RuntimeError("NMFTopicModel must be fitted to access components_.")
        return self.model.components_

    def get_topic_words(self, topic_idx: int, top_n: int = 10) -> list[tuple[str, float]]:
        """Retrieve top keywords and their weights for a given topic.

        Parameters
        ----------
        topic_idx:
            Topic index (0 to n_topics - 1).
        top_n:
            Number of top keywords to return.

        Returns
        -------
        List of (word, weight) tuples sorted descending by weight.
        """
        if not self.is_fitted:
            raise RuntimeError("NMFTopicModel must be fitted before extracting topic words.")
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise ValueError(f"topic_idx {topic_idx} out of range [0, {self.n_topics - 1}].")

        row = self.model.components_[topic_idx]
        total_weight = row.sum()
        norm_weights = row / (total_weight + 1e-12) if total_weight > 0 else row

        top_indices = np.argsort(norm_weights)[::-1][:top_n]
        return [
            (
                self.feature_names[idx] if idx < len(self.feature_names) else f"word_{idx}",
                float(norm_weights[idx]),
            )
            for idx in top_indices
        ]

    def get_topics(self, top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        """Extract top keywords for all discovered topics.

        Parameters
        ----------
        top_n:
            Number of keywords per topic.

        Returns
        -------
        Dictionary mapping topic_id -> list of (word, weight).
        """
        return {k: self.get_topic_words(k, top_n=top_n) for k in range(self.n_topics)}

    def get_document_topics(
        self, X: spmatrix | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get dominant topic, confidence weight, and full topic weights for each document.

        Parameters
        ----------
        X:
            Document-term matrix.

        Returns
        -------
        (dominant_topics, topic_confidences, topic_matrix_W)
        """
        W = self.transform(X)
        # Normalize rows to sum to 1.0 for probabilistic interpretability
        row_sums = W.sum(axis=1, keepdims=True)
        W_norm = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)

        dominant_topics = np.argmax(W, axis=1)
        topic_confidences = np.max(W_norm, axis=1)
        return dominant_topics, topic_confidences, W

    def save(self, path: str | Path) -> None:
        """Serialize fitted NMFTopicModel to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "n_topics": self.n_topics,
                "init": self.init,
                "max_iter": self.max_iter,
                "feature_names": self.feature_names,
                "reconstruction_err_": self.reconstruction_err_,
                "is_fitted": self.is_fitted,
            },
            save_path,
        )
        logger.info(f"Saved NMFTopicModel to {save_path}")

    @classmethod
    def load(cls, path: str | Path) -> "NMFTopicModel":
        """Load serialized NMFTopicModel from disk."""
        data = joblib.load(path)
        instance = cls(
            n_topics=data["n_topics"],
            init=data.get("init", "nndsvda"),
            max_iter=data.get("max_iter", 500),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.reconstruction_err_ = data.get("reconstruction_err_")
        instance.is_fitted = data["is_fitted"]
        logger.info(f"Loaded NMFTopicModel from {path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews
    from reviewiq.representations.tfidf import TfidfVectorizerWrapper

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

    print(f"Vectorizing {len(texts)} reviews with TF-IDF...")
    tfidf = TfidfVectorizerWrapper(max_features=1000)
    X = tfidf.fit_transform(texts)

    nmf = NMFTopicModel(n_topics=4, max_iter=100)
    nmf.fit(X, feature_names=tfidf.feature_names_)

    print("\n--- Discovered NMF Topics ---")
    for t_id, words in nmf.get_topics(top_n=5).items():
        words_str = ", ".join([f"{w} ({weight:.3f})" for w, weight in words])
        print(f"Topic {t_id}: {words_str}")


