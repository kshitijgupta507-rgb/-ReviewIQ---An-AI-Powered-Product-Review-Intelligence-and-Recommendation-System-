"""Amazon Reviews 2023 streaming downloader for ReviewIQ.

Downloads only the laptop-relevant slice of the Electronics dataset.
No need to download the full ~8 GB file.

Strategy
--------
1. Stream the product metadata file (~500 MB compressed).
   Filter rows where ``categories`` contains "Laptop" and NOT any
   exclusion keyword (Accessories, Cases, etc.).
   → Save a set of laptop parent_ASINs and a metadata parquet.

2. Stream the reviews file (~5 GB compressed).
   Keep only rows whose ``parent_asin`` is in the laptop ASIN set.
   → Collect into memory (should be ~200–400 MB of data).

3. Apply per-product sampling:
   - Drop products with < min_reviews reviews.
   - Cap each product at max_reviews reviews (random sample, fixed seed).

4. Save to ``data/interim/laptop_reviews_raw.parquet`` and
   ``data/interim/laptop_meta_raw.parquet``.

Usage
-----
    python -m reviewiq.data.download
    python -m reviewiq.data.download --max-reviews 100   # smaller test sample
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import pathlib
import re
import time
from typing import Iterator

import pandas as pd
import requests
from tqdm import tqdm

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger, setup_logging
from reviewiq.utils.seeds import fix_seeds

logger = get_logger(__name__)

# ── Download URLs ─────────────────────────────────────────────────────────────
# Amazon Reviews 2023 — McAuley Lab, UCSD
_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"
_REVIEWS_URL = f"{_BASE_URL}/review_categories/Electronics.jsonl.gz"
_META_URL = f"{_BASE_URL}/meta_categories/meta_Electronics.jsonl.gz"

# Required columns in output parquet
_REVIEW_COLUMNS = [
    "parent_asin",
    "user_id",
    "rating",
    "title",
    "text",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
]
_META_COLUMNS = [
    "parent_asin",
    "title",
    "brand",
    "price",
    "categories",
    "main_category",
]


def _stream_jsonl_gz(url: str, desc: str) -> Iterator[dict]:
    """Stream a gzipped JSONL file line by line from a URL.

    Yields each line as a parsed dict without loading the whole file
    into memory.

    Parameters
    ----------
    url:
        HTTPS URL to a .jsonl.gz file.
    desc:
        Description shown in the tqdm progress bar.

    Yields
    ------
    dict
        Parsed JSON object for each line.
    """
    logger.info("Streaming %s from %s", desc, url)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to connect to %s: %s", url, e)
        raise

    total_bytes = int(response.headers.get("content-length", 0))
    progress = tqdm(
        total=total_bytes if total_bytes else None,
        unit="B",
        unit_scale=True,
        desc=desc,
        ncols=80,
    )

    buffer = b""
    decompressor = gzip.GzipFile(fileobj=io.BytesIO())

    # Use a streaming decompressor so we never hold >chunk_size in memory
    with gzip.GzipFile(fileobj=_ProgressReader(response, progress)) as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue   # skip malformed lines

    progress.close()


class _ProgressReader:
    """Wraps a requests streaming response to update a tqdm bar as bytes arrive."""

    def __init__(self, response: requests.Response, bar: tqdm) -> None:
        self._iter = response.iter_content(chunk_size=1 << 16)  # 64 KB chunks
        self._bar = bar
        self._buf = b""

    def read(self, n: int = -1) -> bytes:
        out = b""
        while True:
            if n != -1 and len(out) >= n:
                break
            if self._buf:
                rem = n - len(out) if n != -1 else len(self._buf)
                take = self._buf[:rem]
                out += take
                self._buf = self._buf[len(take):]
            else:
                try:
                    chunk = next(self._iter)
                    self._bar.update(len(chunk))
                    self._buf += chunk
                except StopIteration:
                    break
        return out


def _is_laptop(categories: list | str | None, include_kw: list[str], exclude_kw: list[str]) -> bool:
    """Return True if the category list looks like a laptop product.

    Parameters
    ----------
    categories:
        Category list from metadata (can be nested list or string).
    include_kw:
        Keywords that must appear (case-insensitive OR logic).
    exclude_kw:
        Keywords that must NOT appear (case-insensitive OR logic).
    """
    if not categories:
        return False

    # Flatten nested lists or stringify
    if isinstance(categories, list):
        cat_str = " ".join(
            c if isinstance(c, str) else " ".join(c)
            for c in categories
        ).lower()
    else:
        cat_str = str(categories).lower()

    def _matches_any_keyword(keywords: list[str]) -> bool:
        for kw in keywords:
            kw_clean = kw.lower().strip()
            if not kw_clean:
                continue
            forms = [kw_clean]
            if kw_clean.endswith("s") and len(kw_clean) > 3:
                forms.append(kw_clean[:-1])
            pattern = r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b"
            if re.search(pattern, cat_str):
                return True
        return False

    # Must contain at least one include keyword
    if not _matches_any_keyword(include_kw):
        return False
    # Must not contain any exclude keyword
    if _matches_any_keyword(exclude_kw):
        return False
    return True


def download_laptop_metadata(cfg) -> pd.DataFrame:
    """Stream product metadata and return a DataFrame of laptop products.

    Parameters
    ----------
    cfg:
        OmegaConf config (loaded from default.yaml).

    Returns
    -------
    pd.DataFrame
        Metadata for laptop products with columns: parent_asin, title, brand, etc.
    """
    include_kw = list(cfg.data.laptop_include_keywords)
    exclude_kw = list(cfg.data.laptop_exclude_keywords)

    rows = []
    for obj in _stream_jsonl_gz(_META_URL, desc="Metadata"):
        cats = obj.get("categories", [])
        if not _is_laptop(cats, include_kw, exclude_kw):
            continue

        # Flatten categories to a single list of strings
        flat_cats: list[str] = []
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, list):
                    flat_cats.extend(c)
                else:
                    flat_cats.append(str(c))
        else:
            flat_cats = [str(cats)]

        rows.append(
            {
                "parent_asin": obj.get("parent_asin", ""),
                "title": obj.get("title", ""),
                "brand": obj.get("brand", ""),
                "price": obj.get("price"),
                "categories": flat_cats,
                "main_category": obj.get("main_category", ""),
            }
        )

    df = pd.DataFrame(rows, columns=_META_COLUMNS)
    df = df.dropna(subset=["parent_asin"])
    df = df[df["parent_asin"] != ""]
    df = df.drop_duplicates(subset=["parent_asin"])
    logger.info("Found %d laptop products in metadata.", len(df))
    return df


def download_laptop_reviews(
    laptop_asins: set[str],
    cfg,
) -> pd.DataFrame:
    """Stream reviews file and return only laptop reviews.

    Parameters
    ----------
    laptop_asins:
        Set of parent_asin strings to keep.
    cfg:
        OmegaConf config.

    Returns
    -------
    pd.DataFrame
        Reviews for laptop products.
    """
    rows = []
    for obj in _stream_jsonl_gz(_REVIEWS_URL, desc="Reviews"):
        asin = obj.get("parent_asin", "")
        if asin not in laptop_asins:
            continue

        text = obj.get("text", "") or ""
        title = obj.get("title", "") or ""
        if not text.strip():
            continue   # skip empty reviews

        raw_rating = obj.get("rating")
        rating_val = 3.0 if raw_rating is None else raw_rating

        raw_helpful = obj.get("helpful_vote")
        helpful_val = 0 if raw_helpful is None else raw_helpful

        rows.append(
            {
                "parent_asin": asin,
                "user_id": obj.get("user_id", ""),
                "rating": float(rating_val),
                "title": title,
                "text": text,
                "timestamp": obj.get("timestamp", 0),
                "verified_purchase": bool(obj.get("verified_purchase", False)),
                "helpful_vote": int(helpful_val),
            }
        )

    df = pd.DataFrame(rows, columns=_REVIEW_COLUMNS)
    logger.info("Collected %d raw laptop reviews.", len(df))
    return df


def sample_reviews(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply min/max per-product review counts.

    - Drop products with fewer than ``min_reviews_per_product`` reviews.
    - Randomly sample up to ``max_reviews_per_product`` reviews per product.

    Parameters
    ----------
    df:
        Raw reviews DataFrame with ``parent_asin`` column.
    cfg:
        OmegaConf config.

    Returns
    -------
    pd.DataFrame
        Filtered and sampled reviews.
    """
    import numpy as np

    seed = cfg.seed
    min_rev = cfg.data.min_reviews_per_product
    max_rev = cfg.data.max_reviews_per_product

    counts = df.groupby("parent_asin").size()
    valid_asins = counts[counts >= min_rev].index
    df = df[df["parent_asin"].isin(valid_asins)]

    sampled = (
        df.groupby("parent_asin", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_rev), random_state=seed))
    )
    sampled = sampled.reset_index(drop=True)

    n_products = sampled["parent_asin"].nunique()
    logger.info(
        "After sampling: %d reviews across %d products.",
        len(sampled),
        n_products,
    )
    return sampled


def download_laptop_dataset(
    meta_out: str = "data/interim/laptop_meta_raw.parquet",
    reviews_out: str = "data/interim/laptop_reviews_raw.parquet",
    cfg=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full download pipeline: metadata + reviews.

    Parameters
    ----------
    meta_out:
        Output path for metadata parquet.
    reviews_out:
        Output path for reviews parquet.
    cfg:
        Config object. Loaded from default.yaml if None.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (meta_df, reviews_df)
    """
    if cfg is None:
        cfg = load_config()

    fix_seeds(cfg.seed)

    t0 = time.time()

    # Step 1: Metadata
    meta_df = download_laptop_metadata(cfg)
    laptop_asins = set(meta_df["parent_asin"].tolist())

    pathlib.Path(meta_out).parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(meta_out, index=False)
    logger.info("Metadata saved to %s.", meta_out)

    # Step 2: Reviews
    reviews_df = download_laptop_reviews(laptop_asins, cfg)

    # Step 3: Sampling
    reviews_df = sample_reviews(reviews_df, cfg)

    pathlib.Path(reviews_out).parent.mkdir(parents=True, exist_ok=True)
    reviews_df.to_parquet(reviews_out, index=False)
    logger.info("Reviews saved to %s.", reviews_out)

    elapsed = time.time() - t0
    logger.info("Download complete in %.1f seconds.", elapsed)

    return meta_df, reviews_df


# ── CLI ───────────────────────────────────────────────────────────────────────
def _main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Download Amazon laptop reviews")
    parser.add_argument(
        "--meta-out",
        default="data/interim/laptop_meta_raw.parquet",
        help="Output path for metadata parquet",
    )
    parser.add_argument(
        "--reviews-out",
        default="data/interim/laptop_reviews_raw.parquet",
        help="Output path for reviews parquet",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Override max reviews per product (for quick testing)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.max_reviews:
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "data.max_reviews_per_product", args.max_reviews)

    download_laptop_dataset(
        meta_out=args.meta_out,
        reviews_out=args.reviews_out,
        cfg=cfg,
    )


if __name__ == "__main__":
    _main()
