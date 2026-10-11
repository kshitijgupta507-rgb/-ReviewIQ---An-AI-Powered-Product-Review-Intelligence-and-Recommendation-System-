"""Bag of Words (BoW) representation for ReviewIQ.

Wraps scikit-learn's CountVectorizer with support for:
- Both raw review strings and pre-tokenized lists (from PreprocessingPipeline)
- Config-driven parameters (ngram_range, max_features, min_df, max_df)
- Vocabulary inspection and top-feature extraction
- Persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class BagOfWordsVectorizer:
    """Bag-of-Words text vectorizer.

    Parameters
    ----------
    max_features:
        Maximum vocabulary size. If None, read from default config.
    ngram_range:
        Lower and upper boundary of range of n-values for n-grams.
    min_df:
        Minimum document frequency (integer count or float proportion).
    max_df:
        Maximum document frequency (float proportion).
    binary:
        If True, count non-zero occurrences instead of frequencies.
    """

    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] | None = None,
        min_df: int | float | None = None,
        max_df: float | None = None,
        binary: bool = False,
    ) -> None:
        cfg = load_config()
        bow_cfg = getattr(getattr(cfg, "representations", None), "bow", None)

        self.max_features = (
            max_features
            if max_features is not None
            else getattr(bow_cfg, "max_features", 50000)
        )
        self.ngram_range = (
            ngram_range
            if ngram_range is not None
            else tuple(getattr(bow_cfg, "ngram_range", (1, 2)))
        )
        self.min_df = (
            min_df if min_df is not None else getattr(bow_cfg, "min_df", 5)
        )
        self.max_df = (
            max_df if max_df is not None else getattr(bow_cfg, "max_df", 0.95)
        )
        self.binary = binary

        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            binary=self.binary,
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

    def fit(self, corpus: Sequence[str] | Sequence[Sequence[str]]) -> "BagOfWordsVectorizer":
        """Fit the BoW vocabulary on text documents or token lists."""
        prepared = self._prepare_corpus(corpus)
        self.vectorizer.fit(prepared)
        self.is_fitted = True
        logger.info(
            "BoW vocabulary fitted. Vocabulary size: %d terms.",
            len(self.vectorizer.vocabulary_),
        )
        return self

    def transform(self, corpus: Sequence[str] | Sequence[Sequence[str]]) -> spmatrix:
        """Transform text documents or token lists to count matrix."""
        if not self.is_fitted:
            raise ValueError("BagOfWordsVectorizer must be fitted before transforming.")
        prepared = self._prepare_corpus(corpus)
        return self.vectorizer.transform(prepared)

    def fit_transform(
        self, corpus: Sequence[str] | Sequence[Sequence[str]]
    ) -> spmatrix:
        """Fit vocabulary and transform corpus in one pass."""
        prepared = self._prepare_corpus(corpus)
        matrix = self.vectorizer.fit_transform(prepared)
        self.is_fitted = True
        logger.info(
            "BoW fitted and transformed. Shape: %s (vocab: %d).",
            matrix.shape,
            len(self.vectorizer.vocabulary_),
        )
        return matrix

    @property
    def vocabulary(self) -> dict[str, int]:
        """Mapping of terms to feature indices."""
        if not self.is_fitted:
            return {}
        return self.vectorizer.vocabulary_

    def get_feature_names(self) -> list[str]:
        """Return list of vocabulary feature names."""
        if not self.is_fitted:
            return []
        return list(self.vectorizer.get_feature_names_out())

    @property
    def feature_names_(self) -> list[str]:
        """Scikit-learn style property for feature names."""
        return self.get_feature_names()

    def get_top_features(self, doc_vec: spmatrix | np.ndarray, top_k: int = 10) -> list[tuple[str, int]]:
        """Return the top-k most frequent words for a single document vector."""
        if not self.is_fitted:
            return []
        if isinstance(doc_vec, spmatrix):
            doc_arr = doc_vec.toarray().flatten()
        else:
            doc_arr = np.asarray(doc_vec).flatten()

        top_indices = np.argsort(doc_arr)[::-1][:top_k]
        feature_names = self.get_feature_names()
        return [
            (feature_names[i], int(doc_arr[i]))
            for i in top_indices
            if doc_arr[i] > 0
        ]

    def save(self, filepath: str | Path) -> None:
        """Save fitted vectorizer to disk."""
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
                "binary": self.binary,
            },
            path,
        )
        logger.info("Saved BoW vectorizer to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "BagOfWordsVectorizer":
        """Load vectorizer from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No vectorizer artifact found at {path}")
        data = joblib.load(path)
        instance = cls(
            max_features=data.get("max_features"),
            ngram_range=data.get("ngram_range"),
            min_df=data.get("min_df"),
            max_df=data.get("max_df"),
            binary=data.get("binary", False),
        )
        instance.vectorizer = data["vectorizer"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded BoW vectorizer from %s.", path)
        return instance
