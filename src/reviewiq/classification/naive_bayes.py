"""Naive Bayes sentiment classifier for ReviewIQ.

Wraps scikit-learn's MultinomialNB with:
- Config-driven Laplace smoothing (alpha)
- Multi-class sentiment classification (positive, neutral, negative)
- Interpretability: extract top positive, negative, and neutral words via log-probabilities
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.naive_bayes import MultinomialNB

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier for review sentiment.

    Parameters
    ----------
    alpha:
        Additive (Laplace/Lidstone) smoothing parameter. Default from config.
    fit_prior:
        Whether to learn class prior probabilities or assume uniform prior.
    """

    def __init__(
        self,
        alpha: float | None = None,
        fit_prior: bool = True,
    ) -> None:
        cfg = load_config()
        nb_cfg = getattr(getattr(cfg, "classification", None), "naive_bayes", None)

        self.alpha = (
            alpha if alpha is not None else getattr(nb_cfg, "alpha", 1.0)
        )
        self.fit_prior = fit_prior

        self.model = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)
        self.classes_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: Sequence[Any]) -> "NaiveBayesClassifier":
        """Fit Naive Bayes on feature matrix and labels."""
        y_arr = np.asarray(y)
        self.model.fit(X, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        logger.info(
            "Naive Bayes fitted on %d samples across classes: %s.",
            X.shape[0],
            list(self.classes_),
        )
        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict sentiment labels."""
        if not self.is_fitted:
            raise ValueError("NaiveBayesClassifier must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        if not self.is_fitted:
            raise ValueError("NaiveBayesClassifier must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict class log-probabilities."""
        if not self.is_fitted:
            raise ValueError("NaiveBayesClassifier must be fitted before predict_log_proba.")
        return self.model.predict_log_proba(X)

    def top_features_per_class(
        self,
        feature_names: Sequence[str],
        top_n: int = 10,
    ) -> dict[Any, list[tuple[str, float]]]:
        """Extract top most informative words for each sentiment class.

        Uses feature log probabilities (P(word | class)).

        Parameters
        ----------
        feature_names:
            Vocabulary term list matching columns of X.
        top_n:
            Number of top features to return per class.

        Returns
        -------
        dict[Any, list[tuple[str, float]]]
            Mapping from class name to list of (term, log_probability).
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted.")

        feature_names_arr = np.asarray(feature_names)
        results: dict[Any, list[tuple[str, float]]] = {}

        for i, cls_name in enumerate(self.classes_):
            log_probs = self.model.feature_log_prob_[i]
            top_indices = np.argsort(log_probs)[::-1][:top_n]
            results[cls_name] = [
                (str(feature_names_arr[idx]), float(log_probs[idx]))
                for idx in top_indices
            ]

        return results

    def save(self, filepath: str | Path) -> None:
        """Save fitted model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "classes_": self.classes_,
                "alpha": self.alpha,
                "fit_prior": self.fit_prior,
            },
            path,
        )
        logger.info("Saved Naive Bayes model to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "NaiveBayesClassifier":
        """Load fitted Naive Bayes model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        instance = cls(alpha=data.get("alpha"), fit_prior=data.get("fit_prior", True))
        instance.model = data["model"]
        instance.classes_ = data["classes_"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded Naive Bayes model from %s.", path)
        return instance
