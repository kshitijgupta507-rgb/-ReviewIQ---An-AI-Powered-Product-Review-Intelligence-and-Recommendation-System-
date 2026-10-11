"""Unit and integration tests for ReviewIQ Phase 2:
- Feature representations (BoW, TF-IDF, Word2Vec)
- Classifiers (Naive Bayes, Linear SVM, Decision Tree, DistilBERT)
- Model evaluation and benchmark pipeline
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import spmatrix

from reviewiq.classification.decision_tree import DecisionTreeModel
from reviewiq.classification.evaluator import evaluate_classifier, format_comparison_table
from reviewiq.classification.naive_bayes import NaiveBayesClassifier
from reviewiq.classification.svm import SVMClassifier
from reviewiq.classification.benchmark import run_benchmark
from reviewiq.representations.bow import BagOfWordsVectorizer
from reviewiq.representations.tfidf import TfidfVectorizerWrapper
from reviewiq.representations.word2vec import Word2VecWrapper


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_corpus():
    return [
        ["laptop", "battery", "life", "great", "excellent"],
        ["screen", "display", "vibrant", "bright", "good"],
        ["slow", "processor", "overheating", "terrible", "fan"],
        ["cheap", "plastic", "build", "quality", "poor"],
        ["fast", "performance", "smooth", "love", "laptop"],
        ["battery", "drains", "quickly", "awful", "bad"],
    ]


@pytest.fixture
def sample_labels():
    return [
        "positive",
        "positive",
        "negative",
        "negative",
        "positive",
        "negative",
    ]


# ── 1. BoW Tests ──────────────────────────────────────────────────────────────
class TestBagOfWords:
    def test_fit_transform_token_list(self, sample_corpus):
        bow = BagOfWordsVectorizer(min_df=1, max_features=100)
        X = bow.fit_transform(sample_corpus)
        assert isinstance(X, spmatrix)
        assert X.shape[0] == len(sample_corpus)
        assert len(bow.vocabulary) > 0

    def test_fit_transform_string_corpus(self):
        docs = [
            "This laptop has a great battery",
            "Terrible display and poor build quality",
            "Average performance for the price",
        ]
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(docs)
        assert X.shape[0] == 3
        assert "great" in bow.vocabulary
        assert "laptop" in bow.vocabulary

    def test_transform_unfitted_raises(self, sample_corpus):
        bow = BagOfWordsVectorizer()
        with pytest.raises(ValueError, match="must be fitted"):
            bow.transform(sample_corpus)

    def test_get_top_features(self, sample_corpus):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)
        top_words = bow.get_top_features(X[0], top_k=3)
        assert len(top_words) <= 3
        assert all(isinstance(w, str) and count > 0 for w, count in top_words)

    def test_save_and_load(self, sample_corpus):
        bow = BagOfWordsVectorizer(min_df=1)
        X_orig = bow.fit_transform(sample_corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "bow.joblib"
            bow.save(save_path)
            loaded_bow = BagOfWordsVectorizer.load(save_path)

            X_loaded = loaded_bow.transform(sample_corpus)
            assert (X_orig.toarray() == X_loaded.toarray()).all()


# ── 2. TF-IDF Tests ───────────────────────────────────────────────────────────
class TestTfidf:
    def test_fit_transform_shape_and_sublinear(self, sample_corpus):
        tfidf = TfidfVectorizerWrapper(min_df=1, sublinear_tf=True)
        X = tfidf.fit_transform(sample_corpus)
        assert isinstance(X, spmatrix)
        assert X.shape[0] == len(sample_corpus)
        assert len(tfidf.vocabulary) > 0
        assert len(tfidf.idf) == len(tfidf.vocabulary)

    def test_get_top_keywords(self, sample_corpus):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X = tfidf.fit_transform(sample_corpus)
        keywords = tfidf.get_top_keywords(X[0], top_k=3)
        assert len(keywords) <= 3
        assert all(isinstance(word, str) and weight > 0 for word, weight in keywords)

    def test_get_top_keywords_per_class(self, sample_corpus, sample_labels):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X = tfidf.fit_transform(sample_corpus)
        class_kws = tfidf.get_top_keywords_per_class(X, sample_labels, top_k=3)
        assert "positive" in class_kws
        assert "negative" in class_kws
        assert len(class_kws["positive"]) > 0

    def test_save_and_load(self, sample_corpus):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X_orig = tfidf.fit_transform(sample_corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tfidf.joblib"
            tfidf.save(path)
            loaded = TfidfVectorizerWrapper.load(path)
            X_loaded = loaded.transform(sample_corpus)
            assert np.allclose(X_orig.toarray(), X_loaded.toarray())


# ── 3. Word2Vec Tests ─────────────────────────────────────────────────────────
class TestWord2Vec:
    def test_fit_and_transform_mean_pooling(self, sample_corpus):
        w2v = Word2VecWrapper(vector_size=32, min_count=1, epochs=5)
        X = w2v.fit_transform(sample_corpus, aggregation="mean")
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(sample_corpus), 32)
        assert not np.isnan(X).any()

    def test_oov_handling(self, sample_corpus):
        w2v = Word2VecWrapper(vector_size=16, min_count=1)
        w2v.fit(sample_corpus)
        # Vector for unseen out-of-vocabulary word returns zero vector
        oov_vec = w2v.get_vector("completely_unseen_xyz_token")
        assert oov_vec.shape == (16,)
        assert (oov_vec == 0).all()

    def test_most_similar(self, sample_corpus):
        w2v = Word2VecWrapper(vector_size=16, min_count=1)
        w2v.fit(sample_corpus)
        sim = w2v.most_similar("battery", top_k=3)
        assert isinstance(sim, list)
        assert len(sim) <= 3

    def test_save_and_load(self, sample_corpus):
        w2v = Word2VecWrapper(vector_size=16, min_count=1)
        X_orig = w2v.fit_transform(sample_corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "w2v.joblib"
            w2v.save(path)
            loaded = Word2VecWrapper.load(path)
            X_loaded = loaded.transform(sample_corpus)
            assert np.allclose(X_orig, X_loaded, atol=1e-5)


# ── 4. Naive Bayes Tests ──────────────────────────────────────────────────────
class TestNaiveBayes:
    def test_fit_predict_and_proba(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        nb = NaiveBayesClassifier(alpha=1.0)
        nb.fit(X, sample_labels)

        preds = nb.predict(X)
        probs = nb.predict_proba(X)

        assert len(preds) == len(sample_labels)
        assert probs.shape == (len(sample_labels), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_top_features_per_class(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        nb = NaiveBayesClassifier()
        nb.fit(X, sample_labels)

        top_feats = nb.top_features_per_class(bow.get_feature_names(), top_n=3)
        assert "positive" in top_feats
        assert "negative" in top_feats
        assert len(top_feats["positive"]) == 3

    def test_save_and_load(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        nb = NaiveBayesClassifier()
        nb.fit(X, sample_labels)
        preds_orig = nb.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nb.joblib"
            nb.save(path)
            loaded = NaiveBayesClassifier.load(path)
            assert (loaded.predict(X) == preds_orig).all()


# ── 5. SVM Tests ──────────────────────────────────────────────────────────────
class TestSVM:
    def test_fit_predict_and_decision(self, sample_corpus, sample_labels):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X = tfidf.fit_transform(sample_corpus)

        svm = SVMClassifier(C=1.0, calibrate_probabilities=True)
        svm.fit(X, sample_labels)

        preds = svm.predict(X)
        probs = svm.predict_proba(X)

        assert len(preds) == len(sample_labels)
        assert probs.shape[0] == len(sample_labels)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_top_features_coefficients(self, sample_corpus, sample_labels):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X = tfidf.fit_transform(sample_corpus)

        svm = SVMClassifier(C=1.0)
        svm.fit(X, sample_labels)

        top_feats = svm.top_features_per_class(tfidf.get_feature_names(), top_n=3)
        assert len(top_feats) > 0

    def test_save_and_load(self, sample_corpus, sample_labels):
        tfidf = TfidfVectorizerWrapper(min_df=1)
        X = tfidf.fit_transform(sample_corpus)

        svm = SVMClassifier()
        svm.fit(X, sample_labels)
        preds_orig = svm.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "svm.joblib"
            svm.save(path)
            loaded = SVMClassifier.load(path)
            assert (loaded.predict(X) == preds_orig).all()


# ── 6. Decision Tree Tests ────────────────────────────────────────────────────
class TestDecisionTree:
    def test_fit_predict(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        dt = DecisionTreeModel(max_depth=5)
        dt.fit(X, sample_labels)

        preds = dt.predict(X)
        probs = dt.predict_proba(X)

        assert len(preds) == len(sample_labels)
        assert probs.shape == (len(sample_labels), 2)

    def test_top_feature_importances(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        dt = DecisionTreeModel(max_depth=5)
        dt.fit(X, sample_labels)

        importances = dt.top_feature_importances(bow.get_feature_names(), top_n=3)
        assert isinstance(importances, list)
        for term, imp in importances:
            assert isinstance(term, str)
            assert imp >= 0.0

    def test_save_and_load(self, sample_corpus, sample_labels):
        bow = BagOfWordsVectorizer(min_df=1)
        X = bow.fit_transform(sample_corpus)

        dt = DecisionTreeModel()
        dt.fit(X, sample_labels)
        preds_orig = dt.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dt.joblib"
            dt.save(path)
            loaded = DecisionTreeModel.load(path)
            assert (loaded.predict(X) == preds_orig).all()


# ── 7. Evaluator & Benchmark Integration ─────────────────────────────────────
class TestEvaluatorAndBenchmark:
    def test_evaluate_classifier(self, sample_labels):
        preds = ["positive", "positive", "negative", "negative", "positive", "negative"]
        metrics = evaluate_classifier(sample_labels, preds, model_name="TestModel", representation_name="TestRep")

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0
        assert "confusion_matrix" in metrics
        assert "report" in metrics

    def test_format_comparison_table(self):
        results = [
            {
                "representation": "BoW",
                "model": "Naive Bayes",
                "accuracy": 0.85,
                "macro_f1": 0.84,
                "weighted_f1": 0.85,
                "macro_precision": 0.86,
                "macro_recall": 0.83,
            },
            {
                "representation": "TF-IDF",
                "model": "Linear SVM",
                "accuracy": 0.91,
                "macro_f1": 0.90,
                "weighted_f1": 0.91,
                "macro_precision": 0.91,
                "macro_recall": 0.90,
            },
        ]
        table_str = format_comparison_table(results)
        assert "Naive Bayes" in table_str
        assert "Linear SVM" in table_str
        assert "0.9100" in table_str

    def test_run_benchmark(self):
        # Create a balanced toy dataframe
        df = pd.DataFrame(
            {
                "text": [
                    "Great laptop and fast battery",
                    "Brilliant screen and high performance",
                    "Superb build quality and trackpad",
                    "Terrible heating and horrible fan noise",
                    "Broken keyboard and awful processor",
                    "Disappointing battery and cheap plastic",
                    "Decent display and okay keyboard",
                    "Average sound and acceptable weight",
                ] * 4,
                "sentiment_label": [
                    "positive",
                    "positive",
                    "positive",
                    "negative",
                    "negative",
                    "negative",
                    "neutral",
                    "neutral",
                ] * 4,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_df, payload = run_benchmark(df, test_size=0.25, output_dir=tmpdir)
            assert isinstance(summary_df, pd.DataFrame)
            assert len(summary_df) >= 7  # All model/rep combinations evaluated
            assert "Macro F1" in summary_df.columns
            assert "best_model" in payload
            assert (Path(tmpdir) / "sentiment_best_classifier.joblib").exists()
