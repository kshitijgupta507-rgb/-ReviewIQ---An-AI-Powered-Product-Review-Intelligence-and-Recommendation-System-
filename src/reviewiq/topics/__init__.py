"""Topic modeling package for ReviewIQ.

Includes:
- LDATopicModel: Generative probabilistic topic modeling
- NMFTopicModel: Non-negative matrix factorization
- BERTopicModel: Neural topic modeling with sentence embeddings and c-TF-IDF
- Coherence & Diversity evaluation metrics: UMass, NPMI, and benchmarking
"""

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

__all__ = [
    "LDATopicModel",
    "NMFTopicModel",
    "BERTopicModel",
    "calculate_umass_coherence",
    "calculate_npmi_coherence",
    "calculate_topic_diversity",
    "evaluate_topic_model",
    "compare_topic_models",
]
