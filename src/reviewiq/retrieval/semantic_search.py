"""Dense semantic search retriever using FAISS and SentenceTransformers.

Implements dense neural Information Retrieval:
- Encodes review texts into dense vector embeddings using SentenceTransformers
- Normalizes vectors to unit L2-norm so Inner Product search computes exact Cosine Similarity
- Builds FAISS vector index (IndexFlatIP for exact search, or IndexIVFFlat for large corpora)
- Searches dense vector space with sub-millisecond retrieval latency
- Preserves document text and metadata mapping
- Model and index persistence via faiss.write_index / joblib
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import faiss
import joblib
import numpy as np

from reviewiq.retrieval.tfidf_retrieval import SearchResult
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSRetriever:
    """Dense vector retriever using FAISS and SentenceTransformer embeddings.

    Parameters
    ----------
    embedding_model_name:
        Model name for dense embeddings. Default from config `retrieval.embedding_model`
        ('all-MiniLM-L6-v2').
    index_type:
        FAISS index structure: 'flat' (exact IndexFlatIP) or 'ivf' (IndexIVFFlat).
    ivf_nlist:
        Number of clusters for IVF indexing. Default from config (100).
    device:
        Compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        embedding_model_name: str | None = None,
        index_type: str | None = None,
        ivf_nlist: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config()
        r_cfg = getattr(cfg, "retrieval", None)

        self.embedding_model_name = (
            embedding_model_name
            if embedding_model_name is not None
            else getattr(r_cfg, "embedding_model", "all-MiniLM-L6-v2")
        )
        self.index_type = (
            index_type if index_type is not None else getattr(r_cfg, "faiss_index_type", "flat")
        )
        self.ivf_nlist = ivf_nlist if ivf_nlist is not None else getattr(r_cfg, "ivf_nlist", 100)
        self.default_top_k = getattr(r_cfg, "top_k", 20) if r_cfg else 20
        self.device = device

        self._embedder = None
        self.index_: faiss.Index | None = None
        self.dimension: int | None = None
        self.documents: list[str] = []
        self.doc_ids: list[str | int] = []
        self.metadata_list: list[dict[str, Any]] = []
        self.is_indexed = False

    def _get_embedder(self) -> Any:
        """Lazily load SentenceTransformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SentenceTransformer: '{self.embedding_model_name}'...")
            self._embedder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        return self._embedder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Generate unit L2-normalized embeddings for a sequence of texts.

        Normalized vectors allow Inner Product (IP) to correspond exactly to Cosine Similarity.
        """
        embedder = self._get_embedder()
        embeddings = embedder.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _create_index(self, dimension: int, n_samples: int) -> faiss.Index:
        """Construct FAISS index according to index_type."""
        if self.index_type == "ivf" and n_samples >= self.ivf_nlist:
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer, dimension, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT
            )
            return index
        # Default exact flat inner-product index
        return faiss.IndexFlatIP(dimension)

    def index(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str | int] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> "FAISSRetriever":
        """Index a corpus of documents with FAISS.

        Parameters
        ----------
        documents:
            List of review texts.
        doc_ids:
            Optional IDs for documents.
        metadata:
            Optional metadata dict per document.
        embeddings:
            Precomputed normalized embeddings. If None, computes via encode().

        Returns
        -------
        self
        """
        n_docs = len(documents)
        if n_docs == 0:
            raise ValueError("Cannot index an empty document list.")

        logger.info(f"Indexing {n_docs} documents with FAISS...")
        self.documents = list(documents)
        self.doc_ids = (
            list(doc_ids) if doc_ids is not None else [str(i) for i in range(n_docs)]
        )
        self.metadata_list = (
            list(metadata) if metadata is not None else [{} for _ in range(n_docs)]
        )

        if embeddings is None:
            embeddings = self.encode(self.documents)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            # Ensure L2 normalized
            faiss.normalize_L2(embeddings)

        self.dimension = embeddings.shape[1]
        self.index_ = self._create_index(self.dimension, n_docs)

        # Train if needed (e.g. IVF)
        if not self.index_.is_trained:
            logger.info("Training IVF quantizer...")
            self.index_.train(embeddings)

        self.index_.add(embeddings)
        self.is_indexed = True

        logger.info(
            f"FAISS index built successfully. Total indexed vectors: {self.index_.ntotal}, "
            f"dim: {self.dimension}"
        )
        return self

    build_index = index

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search dense index for top semantically relevant documents.

        Parameters
        ----------
        query:
            Natural language search query.
        top_k:
            Number of results to retrieve.
        min_score:
            Score threshold.

        Returns
        -------
        Ranked list of SearchResult objects sorted descending by score.
        """
        if not self.is_indexed or self.index_ is None:
            raise RuntimeError("FAISSRetriever must be indexed before querying.")

        k = top_k if top_k is not None else self.default_top_k
        if not query or not query.strip():
            return []

        # Encode single query vector
        query_vec = self.encode([query])

        # FAISS search
        k_search = min(k, self.index_.ntotal)
        if k_search == 0:
            return []

        scores, indices = self.index_.search(query_vec, k_search)
        scores = scores[0]
        indices = indices[0]

        results: list[SearchResult] = []
        rank = 1
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
            sim_score = float(score)
            if sim_score < min_score:
                continue

            results.append(
                SearchResult(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=sim_score,
                    rank=rank,
                    metadata=self.metadata_list[idx],
                )
            )
            rank += 1

        return results

    def save(self, index_filepath: str | Path, meta_filepath: str | Path | None = None) -> None:
        """Serialize FAISS index and documents metadata to disk."""
        if not self.is_indexed or self.index_ is None:
            raise RuntimeError("Cannot save an unindexed retriever.")

        idx_path = Path(index_filepath)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index_, str(idx_path))

        meta_path = (
            Path(meta_filepath)
            if meta_filepath is not None
            else idx_path.with_suffix(".meta.joblib")
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "embedding_model_name": self.embedding_model_name,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "documents": self.documents,
                "doc_ids": self.doc_ids,
                "metadata_list": self.metadata_list,
                "is_indexed": self.is_indexed,
            },
            meta_path,
        )
        logger.info(f"Saved FAISS index to {idx_path} and metadata to {meta_path}")

    @classmethod
    def load(
        cls,
        index_filepath: str | Path,
        meta_filepath: str | Path | None = None,
        device: str | None = None,
    ) -> "FAISSRetriever":
        """Load serialized FAISS index and metadata from disk."""
        idx_path = Path(index_filepath)
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {idx_path}")

        meta_path = (
            Path(meta_filepath)
            if meta_filepath is not None
            else idx_path.with_suffix(".meta.joblib")
        )
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        meta = joblib.load(meta_path)
        instance = cls(
            embedding_model_name=meta["embedding_model_name"],
            index_type=meta.get("index_type", "flat"),
            device=device,
        )
        instance.index_ = faiss.read_index(str(idx_path))
        instance.dimension = meta["dimension"]
        instance.documents = meta["documents"]
        instance.doc_ids = meta["doc_ids"]
        instance.metadata_list = meta["metadata_list"]
        instance.is_indexed = meta["is_indexed"]

        logger.info(f"Loaded FAISSRetriever with {instance.index_.ntotal} vectors from {idx_path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews

    try:
        df = load_reviews()
        texts = df["review_clean"].tolist() if "review_clean" in df.columns else df["text"].tolist()
        asins = df["parent_asin"].tolist() if "parent_asin" in df.columns else None
        sample_texts = texts[:100]
        sample_asins = asins[:100] if asins else None
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
        ] * 4
        sample_asins = [f"ASIN_{i}" for i in range(len(sample_texts))]

    retriever = FAISSRetriever()
    retriever.index(sample_texts, doc_ids=sample_asins)

    query = "screen brightness and color quality"
    print(f"\n--- Top Results for Query: '{query}' ---")
    results = retriever.search(query, top_k=3)
    for r in results:
        print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.doc_id} | Text: {r.text[:80]}...")
