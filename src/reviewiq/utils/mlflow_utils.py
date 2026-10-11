"""MLflow experiment tracking helpers.

Thin wrappers around mlflow so callers don't need to import mlflow directly
and so the tracking URI is always read from config.

Usage
-----
    from reviewiq.utils.mlflow_utils import start_run, log_metrics, log_params
    with start_run(run_name="nb_baseline"):
        log_params({"alpha": 1.0})
        log_metrics({"macro_f1": 0.82, "accuracy": 0.85})
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import mlflow
from omegaconf import DictConfig

from reviewiq.utils.config import load_config


def init_mlflow(cfg: DictConfig | None = None) -> None:
    """Set MLflow tracking URI and experiment from config.

    Parameters
    ----------
    cfg:
        OmegaConf config object. If None, loads the default config.
    """
    if cfg is None:
        cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


@contextmanager
def start_run(run_name: str, cfg: DictConfig | None = None) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that starts an MLflow run.

    Parameters
    ----------
    run_name:
        Human-readable name for this run (e.g., "svm_tfidf").
    cfg:
        Config object. Initialises MLflow tracking if not already done.

    Yields
    ------
    mlflow.ActiveRun
    """
    init_mlflow(cfg)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log a dictionary of parameters to the active MLflow run."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active MLflow run."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str) -> None:
    """Log a local file or directory as an MLflow artifact."""
    mlflow.log_artifact(local_path)
