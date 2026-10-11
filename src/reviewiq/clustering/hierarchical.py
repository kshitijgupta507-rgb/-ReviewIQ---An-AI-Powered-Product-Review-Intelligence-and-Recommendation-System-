"""Hierarchical clustering for ReviewIQ.

Implements agglomerative hierarchical clustering with Ward linkage:
- Bottom-up clustering producing a cluster hierarchy
- Computes linkage matrix suitable for dendrogram visualization
- Grid search over cluster counts using Silhouette Score to determine optimal k
- Extracts top distinctive keywords per cluster via centroid feature weighting
- Model artifact persistence (save/load via joblib)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse, spmatrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalClusterer:
    """Agglomerative Hierarchical Clustering wrapper.

    Parameters
    ----------
    n_clusters:
        Number of clusters to form. Default 5.
    linkage_type:
        Linkage criterion ('ward', 'complete', 'average', 'single'). Default 'ward'.
    metric:
        Distance metric ('euclidean', 'cosine'). Default 'euclidean' (required for ward).
    """

    def __init__(
        self,
        n_clusters: int | None = 5,
        linkage_type: str | None = None,
        metric: str | None = None,
    ) -> None:
        cfg = load_config()
        c_cfg = getattr(getattr(cfg, "clustering", None), "hierarchical", None)

        self.n_clusters = n_clusters if n_clusters is not None else 5
        self.linkage_type = (
            linkage_type if linkage_type is not None else getattr(c_cfg, "linkage", "ward")
        )
        self.metric = metric if metric is not None else getattr(c_cfg, "metric", "euclidean")

        # Ward linkage requires euclidean distance
        if self.linkage_type == "ward" and self.metric != "euclidean":
            logger.info("Setting metric to 'euclidean' because linkage is 'ward'.")
            self.metric = "euclidean"

        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_type,
            metric=self.metric,
        )
        self.labels_: np.ndarray | None = None
        self.linkage_matrix_: np.ndarray | None = None
        self.is_fitted = False

    @staticmethod
    def _to_dense(X: spmatrix | np.ndarray) -> np.ndarray:
        """Convert sparse matrix to dense array for hierarchical clustering."""
        if issparse(X):
            return X.toarray()
        return np.asarray(X)

    def fit(self, X: spmatrix | np.ndarray) -> "HierarchicalClusterer":
        """Fit agglomerative clustering and assign cluster labels."""
        X_dense = self._to_dense(X)
        self.labels_ = self.model.fit_predict(X_dense)
        self.is_fitted = True
        logger.info(
            "Hierarchical clustering fitted (%d samples, %d clusters, linkage=%s).",
            X_dense.shape[0],
            self.n_clusters,
            self.linkage_type,
        )
        return self

    def compute_linkage_matrix(self, X: spmatrix | np.ndarray) -> np.ndarray:
        """Compute hierarchical linkage matrix for dendrogram generation."""
        X_dense = self._to_dense(X)
        self.linkage_matrix_ = linkage(X_dense, method=self.linkage_type, metric=self.metric)
        return self.linkage_matrix_

    def evaluate_silhouette(self, X: spmatrix | np.ndarray) -> float:
        """Calculate Silhouette Score for current clustering."""
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted before evaluating silhouette score.")
        if len(np.unique(self.labels_)) < 2:
            return 0.0

        X_dense = self._to_dense(X)
        return float(silhouette_score(X_dense, self.labels_, metric=self.metric))

    def find_optimal_clusters(
        self,
        X: spmatrix | np.ndarray,
        k_range: tuple[int, int] | Sequence[int] = (2, 8),
    ) -> tuple[int, dict[int, float]]:
        """Search over candidate cluster counts and return k with highest silhouette score.

        Parameters
        ----------
        X:
            Feature matrix.
        k_range:
            Range or sequence of k values to evaluate.

        Returns
        -------
        tuple[int, dict[int, float]]
            (best_k, dict mapping k -> silhouette_score)
        """
        X_dense = self._to_dense(X)
        n_samples = X_dense.shape[0]

        if isinstance(k_range, tuple) and len(k_range) == 2:
            start_k = max(2, k_range[0])
            candidates = list(range(start_k, min(k_range[1] + 1, n_samples)))
        else:
            candidates = [k for k in k_range if 2 <= k < n_samples]

        if not candidates:
            raise ValueError(
                f"No valid candidate cluster counts k in range {k_range} for n_samples={n_samples}. "
                f"Requires 2 <= k < n_samples."
            )

        scores: dict[int, float] = {}
        best_k = candidates[0]
        best_score = -1.0

        for k in candidates:
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage_type,
                metric=self.metric,
            )
            labels = clusterer.fit_predict(X_dense)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(X_dense, labels, metric=self.metric))
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        logger.info("Optimal clusters search: best_k=%d (silhouette=%.4f).", best_k, best_score)
        return best_k, scores

    def get_cluster_top_terms(
        self,
        X: spmatrix | np.ndarray,
        feature_names: Sequence[str],
        top_n: int = 10,
    ) -> dict[int, list[tuple[str, float]]]:
        """Extract top terms characterizing each cluster by computing cluster centroids.

        Parameters
        ----------
        X:
            Document feature matrix (preferably TF-IDF).
        feature_names:
            Vocabulary term list corresponding to columns of X.
        top_n:
            Number of top keywords to return per cluster.

        Returns
        -------
        dict[int, list[tuple[str, float]]]
            Mapping from cluster ID to list of (term, centroid_weight).
        """
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted.")

        X_dense = self._to_dense(X)
        feature_arr = np.asarray(feature_names)
        results: dict[int, list[tuple[str, float]]] = {}

        for cluster_id in np.unique(self.labels_):
            cluster_mask = self.labels_ == cluster_id
            cluster_samples = X_dense[cluster_mask]
            centroid = np.mean(cluster_samples, axis=0)
            top_indices = np.argsort(centroid)[::-1][:top_n]

            results[int(cluster_id)] = [
                (str(feature_arr[idx]), float(centroid[idx]))
                for idx in top_indices
                if centroid[idx] > 0
            ]

        return results

    # Convenience aliases
    find_optimal_k = find_optimal_clusters
    get_cluster_keywords = get_cluster_top_terms

    def save(self, filepath: str | Path) -> None:
        """Save fitted clusterer to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "labels_": self.labels_,
                "linkage_matrix_": self.linkage_matrix_,
                "n_clusters": self.n_clusters,
                "linkage_type": self.linkage_type,
                "metric": self.metric,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        logger.info("Saved Hierarchical clusterer to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "HierarchicalClusterer":
        """Load fitted clusterer from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")
        data = joblib.load(path)
        instance = cls(
            n_clusters=data.get("n_clusters", 5),
            linkage_type=data.get("linkage_type", "ward"),
            metric=data.get("metric", "euclidean"),
        )
        instance.model = data["model"]
        instance.labels_ = data["labels_"]
        instance.linkage_matrix_ = data.get("linkage_matrix_")
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded Hierarchical clusterer from %s.", path)
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

    clusterer = HierarchicalClusterer(n_clusters=4)
    clusterer.fit(X)

    print(f"Hierarchical Clustering formed {clusterer.n_clusters} clusters.")
    terms = clusterer.get_cluster_top_terms(X, tfidf.feature_names_, top_n=5)
    for c_id, kw in terms.items():
        print(f"Cluster {c_id}: {', '.join([w for w, _ in kw])}")


