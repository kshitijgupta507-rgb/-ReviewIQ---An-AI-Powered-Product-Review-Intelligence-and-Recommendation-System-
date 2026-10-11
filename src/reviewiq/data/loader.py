"""Data loader for ReviewIQ.

Reads parquet files produced by ``download.py`` and validates their
schema against the expected columns. Returns DataFrames with correct
dtypes, ready to feed into the preprocessing pipeline.

Usage
-----
    from reviewiq.data.loader import load_reviews, load_meta
    reviews = load_reviews()
    meta = load_meta()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)

# Expected columns and their dtypes (used for validation)
_REVIEW_SCHEMA: dict[str, str] = {
    "parent_asin": "object",
    "user_id": "object",
    "rating": "float64",
    "title": "object",
    "text": "object",
    "timestamp": "int64",
    "verified_purchase": "bool",
    "helpful_vote": "int64",
}

_META_SCHEMA: dict[str, str] = {
    "parent_asin": "object",
    "title": "object",
    "brand": "object",
}


class DataValidationError(Exception):
    """Raised when a loaded DataFrame is missing required columns."""


def _validate(df: pd.DataFrame, schema: dict[str, str], name: str) -> None:
    """Check that required columns are present.

    Parameters
    ----------
    df:
        DataFrame to validate.
    schema:
        Dict mapping required column names to their expected dtype string.
    name:
        Human-readable name for error messages.

    Raises
    ------
    DataValidationError
        If any required column is missing.
    """
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"{name} is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    logger.debug("%s schema OK. Shape: %s.", name, df.shape)


def load_reviews(
    path: str | Path | None = None,
    cfg=None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the laptop reviews parquet file.

    Parameters
    ----------
    path:
        Path to the reviews parquet file. If None, reads from config.
    cfg:
        OmegaConf config. Loaded from default.yaml if None.
    validate:
        Check schema before returning. Default True.

    Returns
    -------
    pd.DataFrame
        Reviews DataFrame.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist. Run ``download.py`` first.
    DataValidationError
        If required columns are missing.
    """
    if cfg is None:
        cfg = load_config()

    p = Path(path) if path else Path(cfg.data.reviews_parquet)
    if not p.exists():
        raise FileNotFoundError(
            f"Reviews parquet not found at {p}. "
            "Run: python -m reviewiq.data.download"
        )

    df = pd.read_parquet(p)
    logger.info("Loaded %d reviews from %s.", len(df), p)

    if validate:
        _validate(df, _REVIEW_SCHEMA, "Reviews")

    # Coerce dtypes
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)

    return df


def load_meta(
    path: str | Path | None = None,
    cfg=None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the laptop product metadata parquet file.

    Parameters
    ----------
    path:
        Path to the metadata parquet file. If None, reads from config.
    cfg:
        OmegaConf config. Loaded from default.yaml if None.
    validate:
        Check schema before returning. Default True.

    Returns
    -------
    pd.DataFrame
        Product metadata DataFrame.
    """
    if cfg is None:
        cfg = load_config()

    p = Path(path) if path else Path(cfg.data.meta_parquet)
    if not p.exists():
        raise FileNotFoundError(
            f"Metadata parquet not found at {p}. "
            "Run: python -m reviewiq.data.download"
        )

    df = pd.read_parquet(p)
    logger.info("Loaded %d product records from %s.", len(df), p)

    if validate:
        _validate(df, _META_SCHEMA, "Metadata")

    df["brand"] = df["brand"].fillna("Unknown").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    return df


def load_sample(cfg=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the committed small sample data for tests and demos.

    Returns (reviews_df, meta_df) from data/samples/.
    If sample files don't exist, raises FileNotFoundError with instructions.
    """
    if cfg is None:
        cfg = load_config()

    sample_dir = Path(cfg.paths.samples_dir)
    reviews_path = sample_dir / "sample_reviews.parquet"
    meta_path = sample_dir / "sample_meta.parquet"

    missing = [str(p) for p in (reviews_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Sample data missing at {sample_dir}: {', '.join(missing)}. "
            "Please ensure sample files are placed in data/samples/ or generate them with: "
            "python -m reviewiq.data.download --meta-out data/samples/sample_meta.parquet "
            "--reviews-out data/samples/sample_reviews.parquet --max-reviews 30"
        )

    reviews = load_reviews(reviews_path, cfg, validate=False)
    meta = load_meta(meta_path, cfg, validate=False)
    return reviews, meta


def add_sentiment_label(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """Add a ``sentiment_label`` column derived from star rating.

    Mapping:
    - 1–2 stars → "negative"
    - 3 stars   → "neutral"
    - 4–5 stars → "positive"

    Parameters
    ----------
    df:
        Reviews DataFrame with a numeric rating column.
    rating_col:
        Name of the rating column. Default "rating".

    Returns
    -------
    pd.DataFrame
        DataFrame with new ``sentiment_label`` column.
    """
    def _map(r: float) -> str:
        if r <= 2.0:
            return "negative"
        elif r == 3.0:
            return "neutral"
        else:
            return "positive"

    df = df.copy()
    df["sentiment_label"] = df[rating_col].map(_map)
    dist = df["sentiment_label"].value_counts().to_dict()
    logger.info("Sentiment label distribution: %s", dist)
    return df
