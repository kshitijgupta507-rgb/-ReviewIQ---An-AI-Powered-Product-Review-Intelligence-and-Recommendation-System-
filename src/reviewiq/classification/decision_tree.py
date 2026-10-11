"""Decision Tree sentiment classifier for ReviewIQ.

Wraps scikit-learn's DecisionTreeClassifier with:
- Config-controlled max_depth and min_samples_leaf to prevent overfitting
- Gini / entropy impurity splitting
- Feature importance extraction for explainability
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.tree import DecisionTreeClassifier

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionTreeModel:
    """Decision Tree Classifier for sentiment.

    Parameters
    ----------
    max_depth:
        Maximum depth of the tree. Default from config.
    min_samples_leaf:
        Minimum samples required at a leaf node. Default from config.
    criterion:
        Split quality metric ('gini' or 'entropy').
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_leaf: int | None = None,
        criterion: str = "gini",
        random_state: int = 42,
    ) -> None:
        cfg = load_config()
        dt_cfg = getattr(getattr(cfg, "classification", None), "decision_tree", None)

        self.max_depth = (
            max_depth if max_depth is not None else getattr(dt_cfg, "max_depth", 20)
        )
        self.min_samples_leaf = (
            min_samples_leaf
            if min_samples_leaf is not None
            else getattr(dt_cfg, "min_samples_leaf", 5)
        )
        self.criterion = criterion
        self.random_state = random_state

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
        )
        self.classes_: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: spmatrix | np.ndarray, y: Sequence[Any]) -> "DecisionTreeModel":
        """Fit decision tree on feature matrix and labels."""
        y_arr = np.asarray(y)
        self.model.fit(X, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        logger.info(
            "Decision Tree fitted on %d samples (max_depth=%s, n_leaves=%d).",
            X.shape[0],
            self.max_depth,
            self.model.get_n_leaves(),
        )
        return self

    def predict(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict sentiment labels."""
        if not self.is_fitted:
            raise ValueError("DecisionTreeModel must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("DecisionTreeModel must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def top_feature_importances(
        self,
        feature_names: Sequence[str],
        top_n: int = 15,
    ) -> list[tuple[str, float]]:
        """Return the top most important features ranked by Gini importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted.")

        feature_names_arr = np.asarray(feature_names)
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_n]

        return [
            (str(feature_names_arr[idx]), float(importances[idx]))
            for idx in top_indices
            if importances[idx] > 0
        ]

    def save(self, filepath: str | Path) -> None:
        """Save fitted Decision Tree artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "classes_": self.classes_,
                "is_fitted": self.is_fitted,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "criterion": self.criterion,
            },
            path,
        )
        logger.info("Saved Decision Tree model to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "DecisionTreeModel":
        """Load fitted Decision Tree artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        instance = cls(
            max_depth=data.get("max_depth"),
            min_samples_leaf=data.get("min_samples_leaf"),
            criterion=data.get("criterion", "gini"),
        )
        instance.model = data["model"]
        instance.classes_ = data["classes_"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded Decision Tree model from %s.", path)
        return instance
