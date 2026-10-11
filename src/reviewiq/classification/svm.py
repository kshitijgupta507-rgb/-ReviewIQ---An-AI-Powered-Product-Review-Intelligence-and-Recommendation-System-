"""Support Vector Machine (SVM) sentiment classifier for ReviewIQ.

Uses LinearSVC with:
- L2 regularization and squared hinge loss (optimal for sparse text classification)
- Probability calibration (Platt scaling via CalibratedClassifierCV) for predict_proba
- Top-feature coefficient inspection to explain linear decision boundaries
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class SVMClassifier:
    """Linear Support Vector Classifier for sentiment.

    Parameters
    ----------
    C:
        Regularization parameter. Default from config.
    max_iter:
        Maximum number of iterations. Default from config.
    calibrate_probabilities:
        If True, wraps the LinearSVC with CalibratedClassifierCV for
        accurate probability estimation.
    """

    def __init__(
        self,
        C: float | None = None,
        max_iter: int | None = None,
        calibrate_probabilities: bool = True,
        random_state: int = 42,
    ) -> None:
        cfg = load_config()
        svm_cfg = getattr(getattr(cfg, "classification", None), "svm", None)

        self.C = C if C is not None else getattr(svm_cfg, "C", 1.0)
        self.max_iter = (
            max_iter if max_iter is not None else getattr(svm_cfg, "max_iter", 5000)
        )
        self.calibrate_probabilities = calibrate_probabilities
        self.random_state = random_state

        self.base_model = LinearSVC(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.calibrated_model: CalibratedClassifierCV | None = None
        self.classes_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: Sequence[Any]) -> "SVMClassifier":
        """Fit Linear SVM on training features and labels."""
        y_arr = np.asarray(y)

        # Fit base LinearSVC first for feature weights inspection
        self.base_model.fit(X, y_arr)
        self.classes_ = self.base_model.classes_

        if self.calibrate_probabilities:
            # Calibrate probabilities using cross-validation
            n_samples = len(y_arr)
            counts = Counter(y_arr).values()
            min_class_count = min(counts) if counts else 1
            cv_splits = min(3, max(2, min_class_count))
            try:
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state),
                    cv=cv_splits,
                )
                self.calibrated_model.fit(X, y_arr)
            except Exception as e:
                logger.warning("Probability calibration skipped: %s. Using decision_function fallback.", e)
                self.calibrated_model = None

        self.is_fitted = True
        logger.info(
            "SVM fitted on %d samples across classes: %s.",
            X.shape[0],
            list(self.classes_),
        )
        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("SVMClassifier must be fitted before predict.")
        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        return self.base_model.predict(X)

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict calibrated class probabilities."""
        if not self.is_fitted:
            raise ValueError("SVMClassifier must be fitted before predict_proba.")
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)

        # Fallback to normalized decision function
        df = self.base_model.decision_function(X)
        if df.ndim == 1:
            # Binary case
            probs_pos = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        # Softmax for multi-class
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        return exp_df / np.sum(exp_df, axis=1, keepdims=True)

    def decision_function(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Raw decision boundary distances."""
        if not self.is_fitted:
            raise ValueError("SVMClassifier must be fitted before decision_function.")
        return self.base_model.decision_function(X)

    def top_features_per_class(
        self,
        feature_names: Sequence[str],
        top_n: int = 10,
    ) -> dict[Any, list[tuple[str, float]]]:
        """Extract words with highest SVM coefficients per class.

        Parameters
        ----------
        feature_names:
            Vocabulary feature names.
        top_n:
            Number of top features to return.

        Returns
        -------
        dict[Any, list[tuple[str, float]]]
            Top positive weight terms for each class.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted.")

        feature_names_arr = np.asarray(feature_names)
        results: dict[Any, list[tuple[str, float]]] = {}

        if len(self.classes_) == 2:
            # Binary classification: coef_ shape is (1, n_features)
            coefs = self.base_model.coef_[0]
            top_pos_idx = np.argsort(coefs)[::-1][:top_n]
            top_neg_idx = np.argsort(coefs)[:top_n]

            results[self.classes_[1]] = [
                (str(feature_names_arr[idx]), float(coefs[idx]))
                for idx in top_pos_idx
            ]
            results[self.classes_[0]] = [
                (str(feature_names_arr[idx]), float(-coefs[idx]))
                for idx in top_neg_idx
            ]
        else:
            # Multiclass: one-vs-rest coef_ has shape (n_classes, n_features)
            for i, cls_name in enumerate(self.classes_):
                coefs = self.base_model.coef_[i]
                top_idx = np.argsort(coefs)[::-1][:top_n]
                results[cls_name] = [
                    (str(feature_names_arr[idx]), float(coefs[idx]))
                    for idx in top_idx
                ]

        return results

    def save(self, filepath: str | Path) -> None:
        """Save fitted SVM model artifact."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "base_model": self.base_model,
                "calibrated_model": self.calibrated_model,
                "classes_": self.classes_,
                "is_fitted": self.is_fitted,
                "C": self.C,
                "max_iter": self.max_iter,
                "calibrate_probabilities": self.calibrate_probabilities,
            },
            path,
        )
        logger.info("Saved SVM model to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "SVMClassifier":
        """Load fitted SVM model artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        instance = cls(
            C=data.get("C"),
            max_iter=data.get("max_iter"),
            calibrate_probabilities=data.get("calibrate_probabilities", True),
        )
        instance.base_model = data["base_model"]
        instance.calibrated_model = data.get("calibrated_model")
        instance.classes_ = data["classes_"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded SVM model from %s.", path)
        return instance
