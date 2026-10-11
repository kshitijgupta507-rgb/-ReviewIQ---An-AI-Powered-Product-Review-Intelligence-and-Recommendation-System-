"""Evaluation utilities for sentiment classifiers in ReviewIQ.

Computes:
- Overall Accuracy
- Macro-averaged and Weighted Precision, Recall, and F1-score
- Per-class metrics
- Confusion Matrix
- Formats cross-model comparison tables (Markdown)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = get_logger(__name__) if "get_logger" in globals() else logging.getLogger(__name__)


def evaluate_classifier(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    model_name: str = "Classifier",
    representation_name: str = "Feature",
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for a classifier.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_pred:
        Predicted labels.
    model_name:
        Name of classifier algorithm (e.g. 'Naive Bayes', 'Linear SVM').
    representation_name:
        Name of feature representation (e.g. 'BoW', 'TF-IDF', 'Word2Vec').
    labels:
        List of target class labels in specific order.

    Returns
    -------
    dict[str, Any]
        Dictionary of performance metrics and confusion matrix.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if labels is None:
        target_labels = sorted(list(set(y_t) | set(y_p)))
    else:
        target_labels = list(labels)

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=target_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=target_labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_t, y_p, labels=target_labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, labels=target_labels, average="macro", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=target_labels)
    report_dict = classification_report(
        y_t, y_p, labels=target_labels, output_dict=True, zero_division=0
    )

    return {
        "model": model_name,
        "representation": representation_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm,
        "labels": target_labels,
        "report": report_dict,
    }


def format_comparison_table(results: Sequence[dict[str, Any]]) -> str:
    """Format benchmark evaluation metrics into a Markdown comparison table.

    Parameters
    ----------
    results:
        Sequence of evaluation result dictionaries.

    Returns
    -------
    str
        Markdown formatted table.
    """
    headers = [
        "Representation",
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                str(r.get("representation", "N/A")),
                str(r.get("model", "N/A")),
                f"{r.get('accuracy', 0.0):.4f}",
                f"{r.get('macro_f1', 0.0):.4f}",
                f"{r.get('weighted_f1', 0.0):.4f}",
                f"{r.get('macro_precision', 0.0):.4f}",
                f"{r.get('macro_recall', 0.0):.4f}",
            ]
        )

    header_str = "| " + " | ".join(headers) + " |"
    separator_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_str, separator_str, *data_rows])
