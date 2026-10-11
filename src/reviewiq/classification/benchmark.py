"""Benchmarking suite for classical sentiment classifiers in ReviewIQ.

Compares:
- Feature representations: BoW, TF-IDF, Word2Vec
- Classifiers: Multinomial Naive Bayes, Linear SVM, Decision Tree

Outputs a side-by-side performance comparison table (Accuracy, Macro F1, Weighted F1),
and saves the best-performing model artifact for production inference.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from reviewiq.classification.decision_tree import DecisionTreeModel
from reviewiq.classification.evaluator import evaluate_classifier, format_comparison_table
from reviewiq.classification.naive_bayes import NaiveBayesClassifier
from reviewiq.classification.svm import SVMClassifier
from reviewiq.data.loader import add_sentiment_label, load_reviews, load_sample
from reviewiq.preprocessing.pipeline import PreprocessingPipeline
from reviewiq.representations.bow import BagOfWordsVectorizer
from reviewiq.representations.tfidf import TfidfVectorizerWrapper
from reviewiq.representations.word2vec import Word2VecWrapper
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def run_benchmark(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "models",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full matrix evaluation across representations and classifiers.

    Parameters
    ----------
    df:
        DataFrame containing preprocessed reviews with 'clean_tokens' (or 'text')
        and 'sentiment_label' (or 'rating').
    test_size:
        Fraction of samples for testing.
    random_state:
        Seed for train/test split.
    output_dir:
        Directory to save best model and evaluation artifacts.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Comparison summary DataFrame and dictionary of full metric reports.
    """
    if "sentiment_label" not in df.columns:
        if "rating" in df.columns:
            df = add_sentiment_label(df)
        else:
            raise ValueError("Input DataFrame must have 'sentiment_label' or 'rating' column.")

    # Prepare input texts/tokens
    if "clean_tokens" in df.columns:
        corpus = [
            " ".join(tokens) if isinstance(tokens, (list, tuple)) else str(tokens)
            for tokens in df["clean_tokens"]
        ]
    else:
        corpus = [str(t) for t in df["text"].fillna("")]

    y = np.array(list(df["sentiment_label"]), dtype=object)

    # Stratified split
    train_corpus, test_corpus, y_train, y_test = train_test_split(
        corpus,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    logger.info(
        "Dataset split into %d training and %d test samples.",
        len(train_corpus),
        len(test_corpus),
    )

    # 1. Compute Representations
    logger.info("Computing Bag-of-Words features...")
    bow = BagOfWordsVectorizer()
    X_train_bow = bow.fit_transform(train_corpus)
    X_test_bow = bow.transform(test_corpus)

    logger.info("Computing TF-IDF features...")
    tfidf = TfidfVectorizerWrapper()
    X_train_tfidf = tfidf.fit_transform(train_corpus)
    X_test_tfidf = tfidf.transform(test_corpus)

    logger.info("Computing Word2Vec embeddings...")
    w2v = Word2VecWrapper(seed=random_state)
    X_train_w2v = w2v.fit_transform(train_corpus)
    X_test_w2v = w2v.transform(test_corpus)

    # Word2Vec vectors may be negative, so scale to [0, 1] for MultinomialNB
    scaler = MinMaxScaler()
    X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
    X_test_w2v_scaled = scaler.transform(X_test_w2v)

    representations = {
        "BoW": (X_train_bow, X_test_bow),
        "TF-IDF": (X_train_tfidf, X_test_tfidf),
        "Word2Vec": (X_train_w2v, X_test_w2v),
    }

    results = []
    trained_models = {}
    best_f1 = -1.0
    best_model_info = {}

    labels = sorted(list(np.unique(y)))

    # Evaluate each representation with each model
    for rep_name, (X_tr, X_te) in representations.items():
        # A. Naive Bayes
        try:
            nb = NaiveBayesClassifier()
            # Use scaled non-negative features for Word2Vec
            X_tr_nb = X_train_w2v_scaled if rep_name == "Word2Vec" else X_tr
            X_te_nb = X_test_w2v_scaled if rep_name == "Word2Vec" else X_te
            nb.fit(X_tr_nb, y_train)
            y_pred_nb = nb.predict(X_te_nb)
            res_nb = evaluate_classifier(
                y_test, y_pred_nb, model_name="Naive Bayes", representation_name=rep_name, labels=labels
            )
            results.append(res_nb)
            key = f"{rep_name}_NaiveBayes"
            trained_models[key] = (nb, rep_name)
            if res_nb["macro_f1"] > best_f1:
                best_f1 = res_nb["macro_f1"]
                best_model_info = {"key": key, "model": nb, "rep": rep_name, "metrics": res_nb}
        except Exception as e:
            logger.warning("Failed training Naive Bayes on %s: %s", rep_name, e)

        # B. Linear SVM
        try:
            svm = SVMClassifier(random_state=random_state)
            svm.fit(X_tr, y_train)
            y_pred_svm = svm.predict(X_te)
            res_svm = evaluate_classifier(
                y_test, y_pred_svm, model_name="Linear SVM", representation_name=rep_name, labels=labels
            )
            results.append(res_svm)
            key = f"{rep_name}_LinearSVM"
            trained_models[key] = (svm, rep_name)
            if res_svm["macro_f1"] > best_f1:
                best_f1 = res_svm["macro_f1"]
                best_model_info = {"key": key, "model": svm, "rep": rep_name, "metrics": res_svm}
        except Exception as e:
            logger.warning("Failed training SVM on %s: %s", rep_name, e)

        # C. Decision Tree
        try:
            dt = DecisionTreeModel(random_state=random_state)
            dt.fit(X_tr, y_train)
            y_pred_dt = dt.predict(X_te)
            res_dt = evaluate_classifier(
                y_test, y_pred_dt, model_name="Decision Tree", representation_name=rep_name, labels=labels
            )
            results.append(res_dt)
            key = f"{rep_name}_DecisionTree"
            trained_models[key] = (dt, rep_name)
            if res_dt["macro_f1"] > best_f1:
                best_f1 = res_dt["macro_f1"]
                best_model_info = {"key": key, "model": dt, "rep": rep_name, "metrics": res_dt}
        except Exception as e:
            logger.warning("Failed training Decision Tree on %s: %s", rep_name, e)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Save best model and corresponding vectorizer
    if best_model_info:
        best_rep = best_model_info["rep"]
        model_obj = best_model_info["model"]
        model_obj.save(out_p / "sentiment_best_classifier.joblib")

        if best_rep == "BoW":
            bow.save(out_p / "sentiment_best_vectorizer.joblib")
        elif best_rep == "TF-IDF":
            tfidf.save(out_p / "sentiment_best_vectorizer.joblib")
        elif best_rep == "Word2Vec":
            w2v.save(out_p / "sentiment_best_vectorizer.joblib")
            if isinstance(model_obj, NaiveBayesClassifier):
                joblib.dump(scaler, out_p / "sentiment_best_scaler.joblib")

        logger.info(
            "Saved champion model %s (Macro F1: %.4f) to %s.",
            best_model_info["key"],
            best_f1,
            output_dir,
        )

    summary_df = pd.DataFrame(
        [
            {
                "Representation": r["representation"],
                "Model": r["model"],
                "Accuracy": r["accuracy"],
                "Macro F1": r["macro_f1"],
                "Weighted F1": r["weighted_f1"],
                "Macro Precision": r["macro_precision"],
                "Macro Recall": r["macro_recall"],
            }
            for r in results
        ]
    ).sort_values("Macro F1", ascending=False)

    return summary_df, {"results": results, "best_model": best_model_info}


def _main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="ReviewIQ Baseline Sentiment Benchmark")
    parser.add_argument("--use-sample", action="store_true", help="Use small synthetic/sample corpus")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--output-dir", default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    try:
        if args.use_sample:
            logger.info("Loading sample data...")
            reviews_df, _ = load_sample()
        else:
            logger.info("Loading reviews dataset...")
            reviews_df = load_reviews()
    except Exception as e:
        logger.warning("Could not load parquet data (%s). Creating quick inline dataset for test...", e)
        reviews_df = pd.DataFrame(
            {
                "text": [
                    "This laptop is amazing, brilliant screen and wonderful battery!",
                    "Terrible quality, slow processor and broken keyboard.",
                    "Decent everyday machine for light work and web browsing.",
                    "I love the build quality and vibrant colors.",
                    "Worst customer experience, returned immediately after boot failure.",
                    "Average device, nothing particularly special but runs fine.",
                    "Outstanding speed and battery life is very impressive.",
                    "Very disappointed with the loud fan noise and overheating.",
                ] * 10,
                "rating": [5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0] * 10,
            }
        )

    pipe = PreprocessingPipeline(spell_correction=False)
    processed_df = pipe.process_dataframe(reviews_df, show_progress=True)

    summary_df, payload = run_benchmark(
        processed_df, test_size=args.test_size, output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("REVIEWIQ PHASE 2: CLASSICAL CLASSIFICATION BENCHMARK RESULTS")
    print("=" * 80)
    print(format_comparison_table(payload["results"]))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    _main()
