"""Hybrid search combining sparse TF-IDF, dense FAISS, and Cross-Encoder reranking.

Implements state-of-the-art hybrid information retrieval pipeline:
1. Candidate Generation:
   - Sparse lexical retrieval (TF-IDF) capturing exact keyword / spec matches
   - Dense semantic retrieval (FAISS) capturing paraphrases and conceptual meaning
2. Reciprocal Rank Fusion (RRF):
   - Merges ranked lists robustly without requiring fine-tuned score normalization:
     RRF_Score(d) = sum_{m in {sparse, dense}} 1 / (k_rrf + rank_m(d))
3. Cross-Encoder Reranking:
   - Reranks top candidates using deep cross-attention (ms-marco-MiniLM-L-6-v2)
   - Evaluates full query-document interactions to maximize top-k precision
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from reviewiq.retrieval.semantic_search import FAISSRetriever
from reviewiq.retrieval.tfidf_retrieval import SearchResult, TFIDFRetriever
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    ranked_lists:
        List of ranked result lists from different retrieval engines.
    rrf_k:
        Constant smoothing factor (standard default: 60).
    top_k:
        Maximum number of fused results to return.

    Returns
    -------
    Fused list of SearchResult objects ordered descending by RRF score.
    """
    rrf_scores: dict[str | int, float] = defaultdict(float)
    doc_map: dict[str | int, SearchResult] = {}

    for result_list in ranked_lists:
        for item in result_list:
            doc_id = item.doc_id
            rrf_scores[doc_id] += 1.0 / (rrf_k + item.rank)
            # Store doc metadata
            if doc_id not in doc_map:
                doc_map[doc_id] = item

    # Sort descending by fused RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    if top_k is not None:
        sorted_doc_ids = sorted_doc_ids[:top_k]

    fused_results: list[SearchResult] = []
    for rank, doc_id in enumerate(sorted_doc_ids, start=1):
        original = doc_map[doc_id]
        fused_results.append(
            SearchResult(
                doc_id=doc_id,
                text=original.text,
                score=float(rrf_scores[doc_id]),
                rank=rank,
                metadata={**original.metadata, "fusion_method": "rrf"},
            )
        )

    return fused_results


class HybridRetriever:
    """Hybrid Retrieval System combining TF-IDF, FAISS, and Neural Cross-Encoder Reranker.

    Parameters
    ----------
    sparse_retriever:
        Optional preconfigured TFIDFRetriever instance.
    dense_retriever:
        Optional preconfigured FAISSRetriever instance.
    reranker_model_name:
        HuggingFace CrossEncoder model name. Default from config `retrieval.reranker_model`
        ('cross-encoder/ms-marco-MiniLM-L-6-v2').
    rrf_k:
        RRF smoothing constant. Default from config (60).
    candidate_k:
        Number of candidates retrieved from sparse and dense before reranking. Default (20).
    final_k:
        Number of final returned reviews. Default (5).
    device:
        Compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        sparse_retriever: TFIDFRetriever | None = None,
        dense_retriever: FAISSRetriever | None = None,
        reranker_model_name: str | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
        final_k: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config()
        r_cfg = getattr(cfg, "retrieval", None)

        self.sparse_retriever = sparse_retriever or TFIDFRetriever()
        self.dense_retriever = dense_retriever or FAISSRetriever(device=device)

        self.reranker_model_name = (
            reranker_model_name
            if reranker_model_name is not None
            else getattr(r_cfg, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.rrf_k = rrf_k if rrf_k is not None else getattr(r_cfg, "rrf_k", 60)
        self.candidate_k = candidate_k if candidate_k is not None else getattr(r_cfg, "top_k", 20)
        self.final_k = final_k if final_k is not None else getattr(r_cfg, "final_k", 5)
        self.device = device

        self._reranker = None
        self._reranker_load_failed: bool = False
        self.is_indexed = False

    def _get_reranker(self) -> Any:
        """Lazily load CrossEncoder reranking model."""
        if self._reranker_load_failed:
            return None

        if self._reranker is None and self.reranker_model_name:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading CrossEncoder reranker: '{self.reranker_model_name}'...")
                self._reranker = CrossEncoder(
                    self.reranker_model_name, device=self.device
                )
            except Exception as e:
                self._reranker_load_failed = True
                logger.warning(
                    f"Could not load CrossEncoder reranker '{self.reranker_model_name}': {e}. "
                    "Falling back to pure RRF scoring."
                )
                self._reranker = None
        return self._reranker

    def index(
        self,
        documents: Sequence[str],
        doc_ids: Sequence[str | int] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> "HybridRetriever":
        """Index documents into both sparse TF-IDF and dense FAISS engines.

        Parameters
        ----------
        documents:
            List of review texts.
        doc_ids:
            Unique IDs per review.
        metadata:
            Optional metadata dict per review.

        Returns
        -------
        self
        """
        if doc_ids is not None:
            if len(doc_ids) != len(documents):
                raise ValueError(
                    f"Length of doc_ids ({len(doc_ids)}) does not match documents length ({len(documents)})."
                )
            if len(set(doc_ids)) != len(doc_ids):
                raise ValueError("Supplied doc_ids must be unique.")

        logger.info(f"Indexing {len(documents)} documents in HybridRetriever...")
        self.sparse_retriever.index(documents, doc_ids=doc_ids, metadata=metadata)
        self.dense_retriever.index(documents, doc_ids=doc_ids, metadata=metadata)
        self.is_indexed = True
        logger.info("Hybrid indexing complete.")
        return self

    def search(
        self,
        query: str,
        candidate_k: int | None = None,
        final_k: int | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Perform hybrid search: candidate retrieval -> RRF fusion -> cross-encoder rerank.

        Parameters
        ----------
        query:
            Natural language user query.
        candidate_k:
            Number of candidates to gather from each retriever.
        final_k:
            Number of top results to return.
        rerank:
            Whether to execute neural CrossEncoder reranking.

        Returns
        -------
        Final ranked list of SearchResult objects.
        """
        if not self.is_indexed:
            raise RuntimeError("HybridRetriever must be indexed before searching.")

        k_cand = candidate_k if candidate_k is not None else self.candidate_k
        k_final = final_k if final_k is not None else self.final_k

        # 1. Candidate Generation
        sparse_candidates = self.sparse_retriever.search(query, top_k=k_cand)
        dense_candidates = self.dense_retriever.search(query, top_k=k_cand)

        # 2. Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            [sparse_candidates, dense_candidates],
            rrf_k=self.rrf_k,
            top_k=k_cand,
        )

        if not fused_candidates:
            return []

        # 3. Cross-Encoder Reranking
        reranker = self._get_reranker() if rerank else None
        if reranker is None:
            # Return top RRF results directly if no reranker
            return fused_candidates[:k_final]

        logger.debug(f"Reranking top {len(fused_candidates)} candidates with CrossEncoder...")
        pairs = [[query, item.text] for item in fused_candidates]
        cross_scores = reranker.predict(pairs)

        # Sort candidates by neural cross-attention score
        scored_pairs = list(zip(fused_candidates, cross_scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        final_results: list[SearchResult] = []
        for rank, (item, score) in enumerate(scored_pairs[:k_final], start=1):
            final_results.append(
                SearchResult(
                    doc_id=item.doc_id,
                    text=item.text,
                    score=float(score),
                    rank=rank,
                    metadata={**item.metadata, "reranked": True},
                )
            )

        return final_results

    def save(self, directory: str | Path) -> None:
        """Save hybrid retriever components into specified directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.sparse_retriever.save(dir_path / "sparse_tfidf.joblib")
        self.dense_retriever.save(
            dir_path / "dense_faiss.index",
            meta_filepath=dir_path / "dense_faiss.meta.joblib",
        )
        logger.info(f"Saved HybridRetriever to {dir_path}")

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> "HybridRetriever":
        """Load hybrid retriever components from directory."""
        dir_path = Path(directory)
        sparse = TFIDFRetriever.load(dir_path / "sparse_tfidf.joblib")
        dense = FAISSRetriever.load(
            dir_path / "dense_faiss.index",
            meta_filepath=dir_path / "dense_faiss.meta.joblib",
            device=device,
        )
        instance = cls(sparse_retriever=sparse, dense_retriever=dense, device=device)
        instance.is_indexed = True
        logger.info(f"Loaded HybridRetriever from {dir_path}")
        return instance


if __name__ == "__main__":
    from reviewiq.data.loader import load_reviews

    try:
        df = load_reviews()
        sample_df = df.head(100)
        sample_texts = sample_df["review_clean"].tolist() if "review_clean" in sample_df.columns else sample_df["text"].tolist()
        sample_doc_ids = [f"rev_{i}" for i in range(len(sample_texts))]
        sample_metadata = [
            {"parent_asin": row.get("parent_asin", "")} for _, row in sample_df.iterrows()
        ]
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
        sample_doc_ids = [f"rev_{i}" for i in range(len(sample_texts))]
        sample_metadata = [{"parent_asin": f"ASIN_{i % 10}"} for i in range(len(sample_texts))]

    retriever = HybridRetriever()
    retriever.index(sample_texts, doc_ids=sample_doc_ids, metadata=sample_metadata)

    query = "which laptop has the best battery and charging"
    print(f"\n--- Hybrid Search Results for: '{query}' ---")
    results = retriever.search(query, candidate_k=10, final_k=3, rerank=False)
    for r in results:
        print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.doc_id} | Text: {r.text[:80]}...")
