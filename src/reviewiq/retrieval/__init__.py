"""Information retrieval and Named Entity Recognition package for ReviewIQ.

Includes:
- TFIDFRetriever: Sparse lexical search using TF-IDF and cosine similarity
- FAISSRetriever: Dense semantic vector search using FAISS & SentenceTransformers
- HybridRetriever: Multi-stage hybrid search combining RRF fusion & CrossEncoder reranking
- reciprocal_rank_fusion: Reciprocal Rank Fusion algorithm
- SearchResult: Standardized retrieval hit structure
- LaptopNER: Domain-specific Named Entity Recognition for laptop hardware specs
- EntitySpan: Extracted entity span metadata
"""

from reviewiq.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from reviewiq.retrieval.ner import EntitySpan, LaptopNER
from reviewiq.retrieval.semantic_search import FAISSRetriever
from reviewiq.retrieval.tfidf_retrieval import SearchResult, TFIDFRetriever

__all__ = [
    "SearchResult",
    "TFIDFRetriever",
    "FAISSRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "LaptopNER",
    "EntitySpan",
]
