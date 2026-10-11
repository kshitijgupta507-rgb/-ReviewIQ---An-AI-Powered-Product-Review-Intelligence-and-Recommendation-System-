"""DBSCAN density-based clustering for ReviewIQ.

Finds clusters of arbitrary shapes and isolates irrelevant or outlier reviews (marked as noise: label -1).
- Uses cosine or euclidean distance metric (configurable)
- Automatically tracks noise percentage and number of discovered clusters
- Evaluates clustering quality via Silhouette score on non-noise points
- Extracts representative keywords per discovered cluster
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.sparse import issparse, spmatrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class DBSCANClusterer:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps:
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples:
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric:
        The metric to use when calculating distance between instances ('cosine' or 'euclidean').
    """

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int | None = None,
        metric: str | None = None,
    ) -> None:
        cfg = load_config()
        db_cfg = getattr(getattr(cfg, "clustering", None), "dbscan", None)

        self.eps = eps if eps is not None else getattr(db_cfg, "eps", 0.5)
        self.min_samples = (
            min_samples if min_samples is not None else getattr(db_cfg, "min_samples", 5)
        )
        self.metric = metric if metric is not None else getattr(db_cfg, "metric", "cosine")

        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.labels_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.noise_ratio_: float = 0.0
        self.is_fitted = False

    @staticmethod
    def _to_dense(X: spmatrix | np.ndarray) -> np.ndarray:
        """Convert sparse matrix to dense array if necessary."""
        if issparse(X):
            return X.toarray()
        return np.asarray(X)

    def fit(self, X: spmatrix | np.ndarray) -> "DBSCANClusterer":
        """Fit DBSCAN on feature matrix."""
        # For cosine metric on sparse matrices, DBSCAN accepts sparse directly or dense
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True

        unique_labels = set(self.labels_)
        self.n_clusters_ = len(unique_labels - {-1})
        self.noise_ratio_ = float(np.mean(self.labels_ == -1))

        logger.info(
            "DBSCAN fitted on %d samples: %d dense clusters found, noise ratio: %.2f%%.",
            X.shape[0],
            self.n_clusters_,
            self.noise_ratio_ * 100,
        )
        return self

    def evaluate_silhouette(self, X: spmatrix | np.ndarray) -> float:
        """Calculate Silhouette Score excluding noise points (-1)."""
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted before evaluating silhouette score.")

        non_noise_mask = self.labels_ != -1
        non_noise_labels = self.labels_[non_noise_mask]

        if len(np.unique(non_noise_labels)) < 2:
            return 0.0

        if issparse(X):
            X_dense = X[non_noise_mask].toarray()
        else:
            X_dense = np.asarray(X)[non_noise_mask]

        return float(silhouette_score(X_dense, non_noise_labels, metric=self.metric))

    def get_cluster_top_terms(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str],
        top_n: int = 10,
    ) -> dict[int, list[tuple[str, float]]]:
        """Compute average feature weights per cluster to extract representative words."""
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted.")

        X_dense = self._to_dense(X)
        feature_arr = np.asarray(feature_names)
        results: dict[int, list[tuple[str, float]]] = {}

        unique_clusters = sorted([c for c in np.unique(self.labels_) if c != -1])
        for cluster_id in unique_clusters:
            mask = self.labels_ == cluster_id
            cluster_matrix = X_dense[mask]
            centroid = np.mean(cluster_matrix, axis=0)
            top_indices = np.argsort(centroid)[::-1][:top_n]

            results[int(cluster_id)] = [
                (str(feature_arr[idx]), float(centroid[idx]))
                for idx in top_indices
                if centroid[idx] > 0
            ]

        return results

    # Convenience aliases
    evaluate_clustering = evaluate_silhouette
    get_cluster_keywords = get_cluster_top_terms

    def save(self, filepath: str | Path) -> None:
        """Save fitted DBSCAN clusterer to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "labels_": self.labels_,
                "n_clusters_": self.n_clusters_,
                "noise_ratio_": self.noise_ratio_,
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info("Saved DBSCAN clusterer to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "DBSCANClusterer":
        """Load fitted DBSCAN clusterer from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")
        data = joblib.load(path)
        instance = cls(
            eps=data.get("eps"),
            min_samples=data.get("min_samples"),
            metric=data.get("metric"),
        )
        instance.model = data["model"]
        instance.labels_ = data["labels_"]
        instance.n_clusters_ = data["n_clusters_"]
        instance.noise_ratio_ = data["noise_ratio_"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded DBSCAN clusterer from %s.", path)
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews
    from reviewiq.representations.tfidf import TfidfVectorizerWrapper

    try:
        df = load_reviews()
        texts = df["review_clean"].tolist() if "review_clean" in df.columns else df["text"].tolist()
        sample_texts = texts[:300]
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
        ] * 6

    tfidf = TfidfVectorizerWrapper(max_features=500)
    X = tfidf.fit_transform(sample_texts)

    clusterer = DBSCANClusterer(eps=0.8, min_samples=3)
    clusterer.fit(X)

    print(
        f"DBSCAN Clustering: {clusterer.n_clusters_} dense clusters, noise ratio: {clusterer.noise_ratio_ * 100:.1f}%."
    )
    terms = clusterer.get_cluster_top_terms(X, tfidf.feature_names_, top_n=5)
    for c_id, kw in terms.items():
        print(f"Cluster {c_id}: {', '.join([w for w, _ in kw])}")


