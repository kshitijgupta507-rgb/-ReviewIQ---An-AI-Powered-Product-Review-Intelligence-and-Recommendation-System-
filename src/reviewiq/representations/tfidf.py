"""TF-IDF representation for ReviewIQ.

Wraps scikit-learn's TfidfVectorizer with support for:
- Sublinear TF scaling (1 + log(tf)) to prevent dominant terms from swamping
- Both raw review strings and pre-tokenized token lists
- Document-level keyword extraction based on TF-IDF weight
- Class-level keyword aggregation (e.g. positive vs negative keywords)
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class TfidfVectorizerWrapper:
    """TF-IDF vectorizer wrapper.

    Parameters
    ----------
    max_features:
        Maximum vocabulary size. Default loaded from configs/default.yaml.
    ngram_range:
        N-gram range tuple, e.g. (1, 2) for unigrams and bigrams.
    min_df:
        Minimum document frequency.
    max_df:
        Maximum document frequency.
    sublinear_tf:
        Apply sublinear term frequency scaling (1 + log(tf)).
    norm:
        Vector normalization ('l2', 'l1', or None).
    """

    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] | None = None,
        min_df: int | float | None = None,
        max_df: float | None = None,
        sublinear_tf: bool | None = None,
        norm: str | None = "l2",
    ) -> None:
        cfg = load_config()
        tfidf_cfg = getattr(getattr(cfg, "representations", None), "tfidf", None)

        self.max_features = (
            max_features
            if max_features is not None
            else getattr(tfidf_cfg, "max_features", 50000)
        )
        self.ngram_range = (
            ngram_range
            if ngram_range is not None
            else tuple(getattr(tfidf_cfg, "ngram_range", (1, 2)))
        )
        self.min_df = (
            min_df if min_df is not None else getattr(tfidf_cfg, "min_df", 5)
        )
        self.max_df = (
            max_df if max_df is not None else getattr(tfidf_cfg, "max_df", 0.95)
        )
        self.sublinear_tf = (
            sublinear_tf
            if sublinear_tf is not None
            else getattr(tfidf_cfg, "sublinear_tf", True)
        )
        self.norm = norm

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.is_fitted = False

    @staticmethod
    def _prepare_corpus(
        corpus: Sequence[str] | Sequence[Sequence[str]],
    ) -> list[str]:
        """Convert token sequences to space-separated strings if needed."""
        prepared = []
        for item in corpus:
            if isinstance(item, (list, tuple)):
                prepared.append(" ".join(str(token) for token in item))
            elif isinstance(item, str):
                prepared.append(item)
            else:
                prepared.append(str(item))
        return prepared

    def fit(self, corpus: Sequence[str] | Sequence[Sequence[str]]) -> "TfidfVectorizerWrapper":
        """Fit the TF-IDF vocabulary and IDF weights."""
        prepared = self._prepare_corpus(corpus)
        self.vectorizer.fit(prepared)
        self.is_fitted = True
        logger.info(
            "TF-IDF fitted with %d terms (sublinear_tf=%s).",
            len(self.vectorizer.vocabulary_),
            self.sublinear_tf,
        )
        return self

    def transform(self, corpus: Sequence[str] | Sequence[Sequence[str]]) -> spmatrix:
        """Transform documents to normalized TF-IDF matrix."""
        if not self.is_fitted:
            raise ValueError("TfidfVectorizerWrapper must be fitted before transforming.")
        prepared = self._prepare_corpus(corpus)
        return self.vectorizer.transform(prepared)

    def fit_transform(
        self, corpus: Sequence[str] | Sequence[Sequence[str]]
    ) -> spmatrix:
        """Fit and transform in a single pass."""
        prepared = self._prepare_corpus(corpus)
        matrix = self.vectorizer.fit_transform(prepared)
        self.is_fitted = True
        logger.info(
            "TF-IDF fitted & transformed. Shape: %s (vocab: %d).",
            matrix.shape,
            len(self.vectorizer.vocabulary_),
        )
        return matrix

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping of terms to feature column indices."""
        if not self.is_fitted:
            return {}
        return self.vectorizer.vocabulary_

    @property
    def idf(self) -> np.ndarray:
        """Inverse document frequency vector."""
        if not self.is_fitted:
            return np.array([])
        return self.vectorizer.idf_

    def get_feature_names(self) -> list[str]:
        """Return list of vocabulary terms."""
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())

    @property
    def feature_names_(self) -> list[str]:
        """Scikit-learn style property for feature names."""
        return self.get_feature_names()

    def get_top_keywords(
        self, doc_vec: spmatrix | np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Return the top-k highest-weighted keywords for a single document vector."""
        if not self.is_fitted:
            return []
        if isinstance(doc_vec, spmatrix):
            arr = doc_vec.toarray().flatten()
        else:
            arr = np.asarray(doc_vec).flatten()

        top_indices = np.argsort(arr)[::-1][:top_k]
        features = self.get_feature_names()
        return [
            (features[i], float(arr[i]))
            for i in top_indices
            if arr[i] > 0
        ]

    def get_top_keywords_per_class(
        self,
        X: spmatrix,
        y: Sequence[Any],
        top_k: int = 10,
    ) -> dict[Any, list[tuple[str, float]]]:
        """Compute the average TF-IDF weights per class to identify top keywords.

        Parameters
        ----------
        X:
            TF-IDF matrix for a labeled corpus.
        y:
            Class labels corresponding to rows in X.
        top_k:
            Number of top keywords to return per class.

        Returns
        -------
        dict[Any, list[tuple[str, float]]]
            Mapping from class label to list of (term, mean_tfidf_score).
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted.")

        features = self.get_feature_names()
        y_arr = np.asarray(y)
        unique_classes = np.unique(y_arr)
        result: dict[Any, list[tuple[str, float]]] = {}

        for cls in unique_classes:
            mask = y_arr == cls
            cls_matrix = X[mask]
            if cls_matrix.shape[0] == 0:
                result[cls] = []
                continue
            mean_vec = np.asarray(cls_matrix.mean(axis=0)).flatten()
            top_indices = np.argsort(mean_vec)[::-1][:top_k]
            result[cls] = [
                (features[i], float(mean_vec[i]))
                for i in top_indices
                if mean_vec[i] > 0
            ]
        return result

    def save(self, filepath: str | Path) -> None:
        """Save fitted TF-IDF vectorizer to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "is_fitted": self.is_fitted,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "sublinear_tf": self.sublinear_tf,
                "norm": self.norm,
            },
            path,
        )
        logger.info("Saved TF-IDF vectorizer to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "TfidfVectorizerWrapper":
        """Load TF-IDF vectorizer from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No vectorizer artifact found at {path}")
        data = joblib.load(path)
        instance = cls(
            max_features=data.get("max_features"),
            ngram_range=data.get("ngram_range"),
            min_df=data.get("min_df"),
            max_df=data.get("max_df"),
            sublinear_tf=data.get("sublinear_tf", True),
            norm=data.get("norm", "l2"),
        )
        instance.vectorizer = data["vectorizer"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded TF-IDF vectorizer from %s.", path)
        return instance
