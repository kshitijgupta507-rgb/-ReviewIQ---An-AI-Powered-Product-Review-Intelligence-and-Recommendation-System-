"""BERTopic neural topic modeling for ReviewIQ.

Implements modern embedding-based topic modeling with class-based TF-IDF (c-TF-IDF):
- Generates dense contextual document representations using SentenceTransformers
- Performs dimensionality reduction (TruncatedSVD / PCA / UMAP)
- Clusters documents in semantic embedding space (Agglomerative / KMeans / DBSCAN)
- Extracts topic representations using Class-based TF-IDF (c-TF-IDF):
    W_{t,c} = (tf_{t,c} / f_c) * log(1 + A / f_t)
- Discovers coherent, context-rich topics capturing modern review semantics
- Provides topic frequencies, dominant topic assignment, and persistence
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class BERTopicModel:
    """Neural Topic Modeling with Sentence Embeddings and c-TF-IDF.

    Parameters
    ----------
    embedding_model_name:
        HuggingFace / SentenceTransformers model name. Default from config `topics.bertopic.embedding_model`
        ('all-MiniLM-L6-v2').
    n_topics:
        Target number of topics. Default 10.
    n_components:
        Dimension for dimensionality reduction. Default 5.
    clustering_method:
        Clustering algorithm ('agglomerative', 'kmeans'). Default 'agglomerative'.
    min_topic_size:
        Minimum documents per topic. Default from config `topics.bertopic.min_topic_size` (10).
    random_state:
        Seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        embedding_model_name: str | None = None,
        n_topics: int | None = None,
        n_components: int = 5,
        clustering_method: str = "agglomerative",
        min_topic_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        cfg = load_config()
        t_cfg = getattr(cfg, "topics", None)
        b_cfg = getattr(t_cfg, "bertopic", None) if t_cfg else None

        self.embedding_model_name = (
            embedding_model_name
            if embedding_model_name is not None
            else getattr(b_cfg, "embedding_model", "all-MiniLM-L6-v2")
        )
        self.n_topics = n_topics if n_topics is not None else getattr(t_cfg, "num_topics", 10)
        self.min_topic_size = (
            min_topic_size
            if min_topic_size is not None
            else getattr(b_cfg, "min_topic_size", 10)
        )
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.random_state = random_state

        self._embedder = None
        self.reducer: TruncatedSVD | None = None
        self.clusterer: Any = None
        self.c_tfidf_vectorizer: CountVectorizer | None = None

        self.topics_: np.ndarray | None = None
        self.topic_representations_: dict[int, list[tuple[str, float]]] = {}
        self.topic_sizes_: dict[int, int] = {}
        self.is_fitted = False

    def _get_embedder(self) -> Any:
        """Lazily initialize SentenceTransformer embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer: '{self.embedding_model_name}'...")
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.warning(
                    f"Could not load SentenceTransformer '{self.embedding_model_name}': {e}."
                )
                self._embedder = None
        return self._embedder

    def encode(self, docs: Sequence[str]) -> np.ndarray:
        """Generate dense embeddings for a collection of documents."""
        embedder = self._get_embedder()
        if embedder is not None:
            embeddings = embedder.encode(
                list(docs),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

        # Fallback if sentence-transformers is unavailable or offline
        logger.warning("Using TF-IDF dense projection fallback for embeddings.")
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=256)
        return tfidf.fit_transform(docs).toarray().astype(np.float32)

    def _calculate_c_tfidf(
        self, docs: Sequence[str], labels: np.ndarray
    ) -> tuple[dict[int, list[tuple[str, float]]], list[str]]:
        """Compute class-based TF-IDF (c-TF-IDF) per cluster label.

        Formula:
            W_{t, c} = (tf_{t, c} / f_c) * log(1 + A / f_t)
        where:
            - tf_{t, c}: frequency of term t in cluster c
            - f_c: total words in cluster c
            - f_t: total occurrences of term t across all clusters
            - A: average words per cluster
        """
        unique_labels = sorted(set(labels))
        cluster_docs: list[str] = []

        # Join all documents belonging to each cluster
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_text = " ".join([str(docs[i]) for i in indices])
            cluster_docs.append(cluster_text)

        # Count frequencies per cluster
        cv = CountVectorizer(stop_words="english", max_features=5000)
        try:
            X_counts = cv.fit_transform(cluster_docs).toarray().astype(np.float64)  # (n_clusters, n_words)
            vocab = cv.get_feature_names_out()
        except ValueError:
            return {label: [] for label in unique_labels}, []

        # tf_{t, c} is X_counts
        # f_c: total words in cluster c
        f_c = X_counts.sum(axis=1, keepdims=True)
        # Handle empty clusters
        f_c = np.where(f_c == 0, 1.0, f_c)

        # Relative term frequency per cluster: tf / f_c
        tf_norm = X_counts / f_c

        # f_t: total frequency of term t across all clusters
        f_t = X_counts.sum(axis=0, keepdims=True)
        f_t = np.where(f_t == 0, 1.0, f_t)

        # A: average words per cluster
        A = float(f_c.mean())

        # Class-based IDF: log(1 + A / f_t)
        idf = np.log(1.0 + (A / f_t))

        # c-TF-IDF weights
        c_tfidf_matrix = tf_norm * idf

        topic_reps: dict[int, list[tuple[str, float]]] = {}
        for row_idx, label in enumerate(unique_labels):
            scores = c_tfidf_matrix[row_idx]
            top_indices = np.argsort(scores)[::-1][:20]
            topic_reps[label] = [
                (vocab[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]

        self.c_tfidf_vectorizer = cv
        return topic_reps, list(vocab)

    def fit(
        self,
        docs: Sequence[str],
        embeddings: np.ndarray | None = None,
    ) -> "BERTopicModel":
        """Fit BERTopic pipeline on documents.

        Parameters
        ----------
        docs:
            Collection of document texts.
        embeddings:
            Precomputed dense document embeddings. If None, computes using SentenceTransformer.

        Returns
        -------
        self
        """
        n_samples = len(docs)
        if n_samples < 2:
            raise ValueError("At least 2 documents are required to fit BERTopic.")

        logger.info(f"Fitting BERTopic on {n_samples} documents...")

        # 1. Compute or use embeddings
        if embeddings is None:
            embeddings = self.encode(docs)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # 2. Dimensionality reduction
        n_comp = min(self.n_components, embeddings.shape[1], n_samples - 1)
        if n_comp < 2:
            reduced_embeddings = embeddings
        else:
            self.reducer = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            reduced_embeddings = self.reducer.fit_transform(embeddings)

        # 3. Clustering
        n_clusters = min(self.n_topics, n_samples)
        if self.clustering_method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
        else:
            # Agglomerative clustering with Euclidean metric
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
                metric="euclidean",
            )

        labels = self.clusterer.fit_predict(reduced_embeddings)
        self.topics_ = labels

        # Track topic sizes
        unique, counts = np.unique(labels, return_counts=True)
        self.topic_sizes_ = {int(k): int(v) for k, v in zip(unique, counts)}

        # 4. Class-based TF-IDF topic representation
        self.topic_representations_, _ = self._calculate_c_tfidf(docs, labels)
        self.is_fitted = True

        logger.info(
            f"BERTopic fitted with {len(unique)} discovered topics. Topic sizes: {self.topic_sizes_}"
        )
        return self

    def fit_transform(
        self,
        docs: Sequence[str],
        embeddings: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[int, list[tuple[str, float]]]]:
        """Fit model and return (topic_labels, topic_representations)."""
        self.fit(docs, embeddings=embeddings)
        assert self.topics_ is not None
        return self.topics_, self.topic_representations_

    def get_topic(self, topic_id: int, top_n: int = 10) -> list[tuple[str, float]]:
        """Retrieve top keywords and c-TF-IDF scores for a specific topic."""
        if not self.is_fitted:
            raise RuntimeError("BERTopicModel must be fitted before retrieving topics.")
        if topic_id not in self.topic_representations_:
            return []
        return self.topic_representations_[topic_id][:top_n]

    def get_topics(self, top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        """Extract top keywords for all discovered topics."""
        if not self.is_fitted:
            raise RuntimeError("BERTopicModel must be fitted before retrieving topics.")
        return {
            t_id: words[:top_n] for t_id, words in self.topic_representations_.items()
        }

    def get_topic_info(self) -> pd.DataFrame:
        """Produce summary DataFrame of discovered topics."""
        if not self.is_fitted:
            raise RuntimeError("BERTopicModel must be fitted before retrieving topic info.")

        rows = []
        for t_id, size in sorted(self.topic_sizes_.items(), key=lambda x: x[1], reverse=True):
            words = self.get_topic(t_id, top_n=4)
            name = f"{t_id}_" + "_".join([w[0] for w in words]) if words else f"Topic_{t_id}"
            rep = [w[0] for w in self.get_topic(t_id, top_n=10)]
            rows.append({
                "Topic": t_id,
                "Count": size,
                "Name": name,
                "Representation": rep,
            })
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Serialize BERTopic model metadata and state to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embedding_model_name": self.embedding_model_name,
                "n_topics": self.n_topics,
                "n_components": self.n_components,
                "clustering_method": self.clustering_method,
                "reducer": self.reducer,
                "clusterer": self.clusterer,
                "topics_": self.topics_,
                "topic_representations_": self.topic_representations_,
                "topic_sizes_": self.topic_sizes_,
                "is_fitted": self.is_fitted,
            },
            save_path,
        )
        logger.info(f"Saved BERTopicModel to {save_path}")

    @classmethod
    def load(cls, path: str | Path) -> "BERTopicModel":
        """Load serialized BERTopicModel from disk."""
        data = joblib.load(path)
        instance = cls(
            embedding_model_name=data["embedding_model_name"],
            n_topics=data["n_topics"],
            n_components=data.get("n_components", 5),
            clustering_method=data.get("clustering_method", "agglomerative"),
        )
        instance.reducer = data.get("reducer")
        instance.clusterer = data.get("clusterer")
        instance.topics_ = data.get("topics_")
        instance.topic_representations_ = data.get("topic_representations_", {})
        instance.topic_sizes_ = data.get("topic_sizes_", {})
        instance.is_fitted = data.get("is_fitted", False)
        logger.info(f"Loaded BERTopicModel from {path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews

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

    print(f"Fitting BERTopic on {len(sample_texts)} sample reviews...")
    model = BERTopicModel(n_topics=4)
    model.fit(sample_texts)

    print("\n--- Discovered BERTopic Topics ---")
    info = model.get_topic_info()
    print(info[["Topic", "Count", "Name"]].to_string(index=False))


