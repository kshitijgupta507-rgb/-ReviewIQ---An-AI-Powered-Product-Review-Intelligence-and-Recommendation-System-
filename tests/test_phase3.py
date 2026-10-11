"""Unit and integration tests for Phase 3 (Clustering & Topic Modeling).

Covers:
- Hierarchical clustering (Ward linkage, dendrogram matrix, optimal k grid search, centroid keywords, save/load)
- DBSCAN clustering (density clustering, noise detection, silhouette score, keywords, save/load)
- Latent Dirichlet Allocation (document-topic distribution, topic-word distributions, perplexity, save/load)
- Non-negative Matrix Factorization (W and H factorizations, reconstruction error, save/load)
- BERTopic model (c-TF-IDF weighting, topic representations, topic info DataFrame, save/load)
- Topic coherence suite (UMass, NPMI, topic diversity, model comparison table)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from reviewiq.clustering.dbscan import DBSCANClusterer
from reviewiq.clustering.hierarchical import HierarchicalClusterer
from reviewiq.topics.bertopic_model import BERTopicModel
from reviewiq.topics.coherence import (
    calculate_npmi_coherence,
    calculate_topic_diversity,
    calculate_umass_coherence,
    compare_topic_models,
    evaluate_topic_model,
)
from reviewiq.topics.lda import LDATopicModel
from reviewiq.topics.nmf import NMFTopicModel


@pytest.fixture
def corpus_data() -> list[str]:
    """Diverse review sentences representing 3 clear themes: battery, screen/display, build quality."""
    return [
        "Battery life is amazing and lasts all day without charging.",
        "The battery drains super fast and dies in two hours.",
        "Incredible battery performance, great battery standby time.",
        "Battery backup is decent for normal work and browsing.",
        "The screen display is sharp vibrant and colorful with high resolution.",
        "Display is dim and washed out with terrible viewing angles.",
        "OLED screen is gorgeous, deep blacks and great display brightness.",
        "The screen resolution is crisp and beautiful display for movies.",
        "Solid metal build quality, very sturdy hinges and premium chassis.",
        "Cheap plastic build quality, creaky chassis and loose hinges.",
        "Premium aluminum build quality feels sturdy and well constructed.",
        "Flimsy build quality and fragile plastic body broken after a week.",
    ]


@pytest.fixture
def tokenized_corpus(corpus_data) -> list[list[str]]:
    """Tokenized representation of corpus_data."""
    return [doc.lower().replace(".", "").replace(",", "").split() for doc in corpus_data]


# ==============================================================================
# Hierarchical Clustering Tests
# ==============================================================================

class TestHierarchicalClustering:
    def test_fit_and_clusters(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        clusterer = HierarchicalClusterer(n_clusters=3, linkage_type="ward")
        clusterer.fit(X)

        assert clusterer.is_fitted
        assert clusterer.labels_ is not None
        assert len(clusterer.labels_) == len(corpus_data)
        assert len(np.unique(clusterer.labels_)) == 3

    def test_linkage_matrix(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        clusterer = HierarchicalClusterer(n_clusters=3)
        clusterer.fit(X)
        linkage_mat = clusterer.compute_linkage_matrix(X)

        assert linkage_mat is not None
        # Linkage matrix has shape (n_samples - 1, 4)
        assert linkage_mat.shape == (len(corpus_data) - 1, 4)

    def test_optimal_k_search(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        clusterer = HierarchicalClusterer()
        best_k, scores = clusterer.find_optimal_k(X, k_range=[2, 3, 4])

        assert best_k in [2, 3, 4]
        assert len(scores) == 3
        for k in [2, 3, 4]:
            assert k in scores
            assert isinstance(scores[k], float)

    def test_centroid_keywords(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)
        feature_names = vec.get_feature_names_out()

        clusterer = HierarchicalClusterer(n_clusters=3)
        clusterer.fit(X)
        keywords = clusterer.get_cluster_keywords(X, feature_names=feature_names, top_n=3)

        assert len(keywords) == 3
        for c_id, kw_list in keywords.items():
            assert len(kw_list) <= 3
            assert isinstance(kw_list[0], tuple)

    def test_save_load(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        clusterer = HierarchicalClusterer(n_clusters=3)
        clusterer.fit(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "hierarchical.joblib"
            clusterer.save(save_path)
            assert save_path.exists()

            loaded = HierarchicalClusterer.load(save_path)
            assert loaded.is_fitted
            assert loaded.n_clusters == 3
            assert np.array_equal(loaded.labels_, clusterer.labels_)


# ==============================================================================
# DBSCAN Clustering Tests
# ==============================================================================

class TestDBSCANClustering:
    def test_fit_and_noise_tracking(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        dbscan = DBSCANClusterer(eps=0.8, min_samples=2, metric="cosine")
        dbscan.fit(X)

        assert dbscan.is_fitted
        assert dbscan.labels_ is not None
        assert len(dbscan.labels_) == len(corpus_data)
        assert 0.0 <= dbscan.noise_ratio_ <= 1.0

    def test_silhouette_score_evaluation(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        dbscan = DBSCANClusterer(eps=0.8, min_samples=2, metric="cosine")
        dbscan.fit(X)
        score = dbscan.evaluate_clustering(X)

        if dbscan.n_clusters_ >= 2:
            assert score is not None
            assert -1.0 <= score <= 1.0

    def test_cluster_keywords(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)
        feature_names = vec.get_feature_names_out()

        dbscan = DBSCANClusterer(eps=0.9, min_samples=2, metric="cosine")
        dbscan.fit(X)
        keywords = dbscan.get_cluster_keywords(X, feature_names=feature_names, top_n=3)

        assert isinstance(keywords, dict)
        for c_id, kw_list in keywords.items():
            assert c_id != -1  # Noise is excluded from topic cluster keywords
            assert len(kw_list) <= 3

    def test_save_load(self, corpus_data):
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus_data)

        dbscan = DBSCANClusterer(eps=0.7, min_samples=2)
        dbscan.fit(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "dbscan.joblib"
            dbscan.save(save_path)
            assert save_path.exists()

            loaded = DBSCANClusterer.load(save_path)
            assert loaded.is_fitted
            assert loaded.eps == dbscan.eps
            assert np.array_equal(loaded.labels_, dbscan.labels_)


# ==============================================================================
# LDA Topic Model Tests
# ==============================================================================

class TestLDATopicModel:
    def test_fit_and_transform(self, corpus_data):
        cv = CountVectorizer(stop_words="english")
        X = cv.fit_transform(corpus_data)
        vocab = cv.get_feature_names_out()

        lda = LDATopicModel(n_topics=3, max_iter=20, random_state=42)
        doc_topics = lda.fit_transform(X, feature_names=vocab)

        assert lda.is_fitted
        assert doc_topics.shape == (len(corpus_data), 3)
        # Probabilities should sum to approximately 1.0 per document
        np.testing.assert_allclose(doc_topics.sum(axis=1), np.ones(len(corpus_data)), atol=1e-4)

    def test_topic_words(self, corpus_data):
        cv = CountVectorizer(stop_words="english")
        X = cv.fit_transform(corpus_data)
        vocab = cv.get_feature_names_out()

        lda = LDATopicModel(n_topics=3, max_iter=20, random_state=42)
        lda.fit(X, feature_names=vocab)

        topics = lda.get_topics(top_n=5)
        assert len(topics) == 3
        for t_id, words in topics.items():
            assert len(words) == 5
            total_prob = sum(prob for _, prob in words)
            assert total_prob > 0.0

    def test_document_topics(self, corpus_data):
        cv = CountVectorizer(stop_words="english")
        X = cv.fit_transform(corpus_data)

        lda = LDATopicModel(n_topics=3, max_iter=15, random_state=42)
        lda.fit(X)

        dom_topics, confs, dist = lda.get_document_topics(X)
        assert len(dom_topics) == len(corpus_data)
        assert len(confs) == len(corpus_data)
        assert np.all(dom_topics >= 0) and np.all(dom_topics < 3)
        assert np.all(confs >= 0.0) and np.all(confs <= 1.0)

    def test_perplexity_and_score(self, corpus_data):
        cv = CountVectorizer(stop_words="english")
        X = cv.fit_transform(corpus_data)

        lda = LDATopicModel(n_topics=3, max_iter=15, random_state=42)
        lda.fit(X)

        perp = lda.perplexity(X)
        score = lda.score(X)
        assert isinstance(perp, float) and perp > 0
        assert isinstance(score, float) and score < 0  # log likelihood is negative

    def test_save_load(self, corpus_data):
        cv = CountVectorizer(stop_words="english")
        X = cv.fit_transform(corpus_data)

        lda = LDATopicModel(n_topics=2, max_iter=10, random_state=42)
        lda.fit(X, feature_names=cv.get_feature_names_out())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lda.joblib"
            lda.save(save_path)
            assert save_path.exists()

            loaded = LDATopicModel.load(save_path)
            assert loaded.is_fitted
            assert loaded.n_topics == 2
            assert loaded.feature_names == lda.feature_names


# ==============================================================================
# NMF Topic Model Tests
# ==============================================================================

class TestNMFTopicModel:
    def test_fit_and_transform(self, corpus_data):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)

        nmf = NMFTopicModel(n_topics=3, max_iter=100, random_state=42)
        W = nmf.fit_transform(X, feature_names=vec.get_feature_names_out())

        assert nmf.is_fitted
        assert W.shape == (len(corpus_data), 3)
        assert np.all(W >= 0.0)  # Non-negative weights

    def test_reconstruction_and_components(self, corpus_data):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)

        nmf = NMFTopicModel(n_topics=3, max_iter=100, random_state=42)
        nmf.fit(X, feature_names=vec.get_feature_names_out())

        assert nmf.reconstruction_err_ is not None
        assert nmf.reconstruction_err_ >= 0.0
        H = nmf.components_
        assert H.shape == (3, X.shape[1])
        assert np.all(H >= 0.0)

    def test_topic_words(self, corpus_data):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)

        nmf = NMFTopicModel(n_topics=3, max_iter=100, random_state=42)
        nmf.fit(X, feature_names=vec.get_feature_names_out())

        topics = nmf.get_topics(top_n=4)
        assert len(topics) == 3
        for t_id, kw in topics.items():
            assert len(kw) == 4
            assert kw[0][1] >= kw[1][1]  # Sorted descending

    def test_document_topics(self, corpus_data):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)

        nmf = NMFTopicModel(n_topics=3, max_iter=100, random_state=42)
        nmf.fit(X)

        dom_topics, confs, W = nmf.get_document_topics(X)
        assert len(dom_topics) == len(corpus_data)
        assert len(confs) == len(corpus_data)
        assert np.all(dom_topics >= 0) and np.all(dom_topics < 3)

    def test_save_load(self, corpus_data):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)

        nmf = NMFTopicModel(n_topics=2, max_iter=50, random_state=42)
        nmf.fit(X, feature_names=vec.get_feature_names_out())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nmf.joblib"
            nmf.save(save_path)
            assert save_path.exists()

            loaded = NMFTopicModel.load(save_path)
            assert loaded.is_fitted
            assert loaded.n_topics == 2
            assert loaded.feature_names == nmf.feature_names


# ==============================================================================
# BERTopic Model Tests
# ==============================================================================

class TestBERTopicModel:
    def test_fit_and_representations(self, corpus_data):
        # Using synthetic mock dense embeddings to test rapid clustering & c-TF-IDF
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(corpus_data), 32).astype(np.float32)

        bertopic = BERTopicModel(n_topics=3, n_components=4, random_state=42)
        labels, reps = bertopic.fit_transform(corpus_data, embeddings=mock_embeddings)

        assert bertopic.is_fitted
        assert len(labels) == len(corpus_data)
        assert len(reps) <= 3
        topics = bertopic.get_topics(top_n=3)
        assert len(topics) > 0

    def test_topic_info_dataframe(self, corpus_data):
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(corpus_data), 16).astype(np.float32)

        bertopic = BERTopicModel(n_topics=3, n_components=4, random_state=42)
        bertopic.fit(corpus_data, embeddings=mock_embeddings)

        info_df = bertopic.get_topic_info()
        assert isinstance(info_df, pd.DataFrame)
        assert "Topic" in info_df.columns
        assert "Count" in info_df.columns
        assert "Name" in info_df.columns
        assert "Representation" in info_df.columns
        assert len(info_df) > 0

    def test_save_load(self, corpus_data):
        np.random.seed(42)
        mock_embeddings = np.random.randn(len(corpus_data), 16).astype(np.float32)

        bertopic = BERTopicModel(n_topics=2, n_components=3, random_state=42)
        bertopic.fit(corpus_data, embeddings=mock_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "bertopic.joblib"
            bertopic.save(save_path)
            assert save_path.exists()

            loaded = BERTopicModel.load(save_path)
            assert loaded.is_fitted
            assert loaded.n_topics == 2
            assert loaded.topic_sizes_ == bertopic.topic_sizes_


# ==============================================================================
# Coherence and Diversity Evaluation Tests
# ==============================================================================

class TestTopicCoherence:
    def test_umass_coherence(self, tokenized_corpus):
        # Battery-focused keywords that co-occur together
        topic_words = ["battery", "life", "drains", "hours", "standby"]
        score = calculate_umass_coherence(topic_words, tokenized_corpus, top_n=5)
        assert isinstance(score, float)
        assert score <= 0.0  # UMass score is typically negative

    def test_npmi_coherence(self, tokenized_corpus):
        topic_words = ["screen", "display", "resolution", "vibrant", "sharp"]
        score = calculate_npmi_coherence(topic_words, tokenized_corpus, top_n=5)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_topic_diversity(self):
        diverse_topics = {
            0: [("battery", 0.5), ("charger", 0.4), ("standby", 0.3)],
            1: [("screen", 0.6), ("display", 0.4), ("brightness", 0.2)],
            2: [("hinges", 0.7), ("metal", 0.3), ("chassis", 0.1)],
        }
        redundant_topics = {
            0: [("battery", 0.5), ("screen", 0.4), ("display", 0.3)],
            1: [("battery", 0.6), ("screen", 0.4), ("display", 0.2)],
        }

        diverse_score = calculate_topic_diversity(diverse_topics, top_n=3)
        redundant_score = calculate_topic_diversity(redundant_topics, top_n=3)

        assert diverse_score == 1.0  # 9 unique words / 9 total words
        assert redundant_score < 1.0

    def test_evaluate_topic_model(self, tokenized_corpus):
        topics = {
            0: [("battery", 0.5), ("life", 0.3)],
            1: [("screen", 0.4), ("display", 0.2)],
        }
        metrics = evaluate_topic_model(topics, tokenized_corpus, top_n=2)

        assert "umass_coherence" in metrics
        assert "npmi_coherence" in metrics
        assert "topic_diversity" in metrics
        assert isinstance(metrics["umass_coherence"], float)
        assert isinstance(metrics["npmi_coherence"], float)
        assert isinstance(metrics["topic_diversity"], float)

    def test_compare_topic_models(self, corpus_data, tokenized_corpus):
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(corpus_data)
        cv = CountVectorizer(stop_words="english")
        X_counts = cv.fit_transform(corpus_data)

        lda = LDATopicModel(n_topics=2, max_iter=10, random_state=42)
        lda.fit(X_counts, feature_names=cv.get_feature_names_out())

        nmf = NMFTopicModel(n_topics=2, max_iter=50, random_state=42)
        nmf.fit(X, feature_names=vec.get_feature_names_out())

        models = {"LDA": lda, "NMF": nmf}
        df_comparison = compare_topic_models(models, tokenized_corpus, top_n=3)

        assert isinstance(df_comparison, pd.DataFrame)
        assert len(df_comparison) == 2
        assert "Model" in df_comparison.columns
        assert "UMass Coherence" in df_comparison.columns
        assert "NPMI Coherence" in df_comparison.columns
        assert "Topic Diversity" in df_comparison.columns
