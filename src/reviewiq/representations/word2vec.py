"""Word2Vec representation for ReviewIQ.

Learns dense, continuous distributed semantic representations of words
trained directly on the laptop review corpus. Captures domain-specific
relationships (e.g. "thermals", "throttling", "bloatware").

Implementation:
- Uses gensim's Word2Vec when available.
- Features a robust pure-NumPy continuous embedding fallback when gensim C-extensions
  are unavailable (e.g., Python 3.14 on Windows without MSVC C++ build tools).
- Computes document-level representations using mean pooling or TF-IDF weighted pooling.
- Supports nearest-neighbor query (`most_similar`) and model persistence.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from gensim.models import Word2Vec as GensimWord2Vec

    _GENSIM_AVAILABLE = True
except ImportError:
    _GENSIM_AVAILABLE = False


class _NumpyWord2Vec:
    """Pure-NumPy Skip-gram with negative sampling / co-occurrence embedding fallback."""

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        learning_rate: float = 0.025,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed

        self.wv: dict[str, np.ndarray] = {}
        self.vocab: dict[str, int] = {}
        self.index_to_key: list[str] = []

    def fit(self, sentences: Sequence[Sequence[str]]) -> None:
        rng = np.random.RandomState(self.seed)

        # Build vocabulary
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence)

        self.index_to_key = [
            word for word, count in counts.items() if count >= self.min_count
        ]
        self.vocab = {word: i for i, word in enumerate(self.index_to_key)}
        vocab_size = len(self.index_to_key)

        if vocab_size == 0:
            logger.warning("Vocabulary is empty after applying min_count=%d", self.min_count)
            return

        # Initialize input and output embedding weights
        W_in = (rng.rand(vocab_size, self.vector_size) - 0.5) / self.vector_size
        W_out = (rng.rand(vocab_size, self.vector_size) - 0.5) / self.vector_size

        # Training pairs generation with Skip-Gram and Negative Sampling
        pairs: list[tuple[int, int]] = []
        for sentence in sentences:
            indices = [self.vocab[w] for w in sentence if w in self.vocab]
            for i, target in enumerate(indices):
                start = max(0, i - self.window)
                end = min(len(indices), i + self.window + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((target, indices[j]))

        if not pairs:
            # Fallback random normalized vectors if corpus is too small
            for i, word in enumerate(self.index_to_key):
                vec = W_in[i]
                norm = np.linalg.norm(vec)
                self.wv[word] = vec / (norm if norm > 0 else 1.0)
            return

        pairs_arr = np.array(pairs, dtype=np.int32)
        lr = self.learning_rate

        for _ in range(self.epochs):
            rng.shuffle(pairs_arr)
            # Sample a subset for speed if large
            sample_pairs = pairs_arr[: min(len(pairs_arr), 100000)]
            for target_idx, context_idx in sample_pairs:
                v_target = W_in[target_idx]
                v_context = W_out[context_idx]

                # Positive pair sigmoid
                score = np.dot(v_target, v_context)
                sig = 1.0 / (1.0 + np.exp(-np.clip(score, -10.0, 10.0)))
                err = sig - 1.0

                # Negative sample
                neg_idx = rng.randint(0, vocab_size)
                v_neg = W_out[neg_idx]
                score_neg = np.dot(v_target, v_neg)
                sig_neg = 1.0 / (1.0 + np.exp(-np.clip(score_neg, -10.0, 10.0)))
                err_neg = sig_neg

                # Gradients update
                grad_target = err * v_context + err_neg * v_neg
                W_in[target_idx] -= lr * grad_target
                W_out[context_idx] -= lr * err * v_target
                W_out[neg_idx] -= lr * err_neg * v_target

        for i, word in enumerate(self.index_to_key):
            vec = W_in[i]
            norm = np.linalg.norm(vec)
            self.wv[word] = vec / (norm if norm > 0 else 1.0)

    def __contains__(self, word: str) -> bool:
        return word in self.wv

    def get_vector(self, word: str) -> np.ndarray:
        return self.wv[word]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if word not in self.wv:
            raise KeyError(f"Word '{word}' not in vocabulary.")

        target_vec = self.wv[word]
        scores = []
        for other_word, vec in self.wv.items():
            if other_word == word:
                continue
            cos = float(np.dot(target_vec, vec))
            scores.append((other_word, cos))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]


class Word2VecWrapper:
    """ReviewIQ Word2Vec embeddings and document-level aggregator.

    Parameters
    ----------
    vector_size:
        Embedding dimensions (e.g. 100 or 300).
    window:
        Maximum distance between current and predicted word.
    min_count:
        Ignores words with total frequency lower than this.
    epochs:
        Number of training iterations over the corpus.
    sg:
        Training algorithm: 1 for Skip-gram; 0 for CBOW.
    workers:
        Number of worker threads (gensim).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        vector_size: int | None = None,
        window: int | None = None,
        min_count: int | None = None,
        epochs: int | None = None,
        sg: int | None = None,
        workers: int | None = None,
        seed: int = 42,
    ) -> None:
        cfg = load_config()
        w2v_cfg = getattr(getattr(cfg, "representations", None), "word2vec", None)

        self.vector_size = (
            vector_size
            if vector_size is not None
            else getattr(w2v_cfg, "vector_size", 100)
        )
        self.window = (
            window if window is not None else getattr(w2v_cfg, "window", 5)
        )
        self.min_count = (
            min_count if min_count is not None else getattr(w2v_cfg, "min_count", 2)
        )
        self.epochs = (
            epochs if epochs is not None else getattr(w2v_cfg, "epochs", 5)
        )
        self.sg = sg if sg is not None else getattr(w2v_cfg, "sg", 1)
        self.workers = (
            workers if workers is not None else getattr(w2v_cfg, "workers", 4)
        )
        self.seed = seed

        self.model: Any = None
        self.is_fitted = False

    @staticmethod
    def _prepare_corpus(
        corpus: Sequence[str] | Sequence[Sequence[str]],
    ) -> list[list[str]]:
        """Normalize documents to list of token lists."""
        tokenized = []
        for doc in corpus:
            if isinstance(doc, str):
                tokenized.append(doc.lower().split())
            elif isinstance(doc, (list, tuple)):
                tokenized.append([str(t).lower() for t in doc])
            else:
                tokenized.append(str(doc).lower().split())
        return tokenized

    def fit(self, corpus: Sequence[str] | Sequence[Sequence[str]]) -> "Word2VecWrapper":
        """Train Word2Vec embeddings on the tokenized corpus."""
        sentences = self._prepare_corpus(corpus)

        if _GENSIM_AVAILABLE:
            logger.info("Training Word2Vec via gensim (size=%d, sg=%d)...", self.vector_size, self.sg)
            self.model = GensimWord2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                sg=self.sg,
                workers=self.workers,
                seed=self.seed,
            )
        else:
            logger.info("Training Word2Vec via NumPy fallback (size=%d)...", self.vector_size)
            numpy_model = _NumpyWord2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs,
                seed=self.seed,
            )
            numpy_model.fit(sentences)
            self.model = numpy_model

        self.is_fitted = True
        return self

    def get_vector(self, word: str) -> np.ndarray:
        """Return embedding vector for a given word."""
        if not self.is_fitted:
            raise ValueError("Word2VecWrapper must be fitted.")
        word = word.lower()
        if _GENSIM_AVAILABLE and isinstance(self.model, GensimWord2Vec):
            if word in self.model.wv:
                return np.asarray(self.model.wv[word])
        elif hasattr(self.model, "wv") and word in self.model:
            return np.asarray(self.model.get_vector(word))

        return np.zeros(self.vector_size, dtype=np.float32)

    def most_similar(self, word: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return most semantically similar words to target."""
        if not self.is_fitted:
            raise ValueError("Word2VecWrapper must be fitted.")
        word = word.lower()

        if _GENSIM_AVAILABLE and isinstance(self.model, GensimWord2Vec):
            if word not in self.model.wv:
                raise KeyError(f"Word '{word}' not in Word2Vec vocabulary.")
            return [
                (w, float(score))
                for w, score in self.model.wv.most_similar(word, topn=top_k)
            ]
        elif hasattr(self.model, "most_similar"):
            return self.model.most_similar(word, topn=top_k)

        return []

    def transform(
        self,
        corpus: Sequence[str] | Sequence[Sequence[str]],
        aggregation: str = "mean",
    ) -> np.ndarray:
        """Convert documents into dense vectors via word vector pooling.

        Parameters
        ----------
        corpus:
            Collection of text documents or token lists.
        aggregation:
            Pooling strategy: 'mean' (default) or 'sum'.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_documents, vector_size).
        """
        if not self.is_fitted:
            raise ValueError("Word2VecWrapper must be fitted before transforming.")

        sentences = self._prepare_corpus(corpus)
        n_docs = len(sentences)
        embeddings = np.zeros((n_docs, self.vector_size), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            vecs = []
            for token in sentence:
                if _GENSIM_AVAILABLE and isinstance(self.model, GensimWord2Vec):
                    if token in self.model.wv:
                        vecs.append(self.model.wv[token])
                elif hasattr(self.model, "wv") and token in self.model:
                    vecs.append(self.model.get_vector(token))

            if vecs:
                arr = np.array(vecs, dtype=np.float32)
                if aggregation == "sum":
                    embeddings[i] = np.sum(arr, axis=0)
                else:  # 'mean'
                    embeddings[i] = np.mean(arr, axis=0)
            # else leaves zero vector for empty / entirely OOV documents

        return embeddings

    def fit_transform(
        self,
        corpus: Sequence[str] | Sequence[Sequence[str]],
        aggregation: str = "mean",
    ) -> np.ndarray:
        """Fit model and return document vector matrix."""
        self.fit(corpus)
        return self.transform(corpus, aggregation=aggregation)

    def save(self, filepath: str | Path) -> None:
        """Save fitted Word2Vec model artifact."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "vector_size": self.vector_size,
                "window": self.window,
                "min_count": self.min_count,
                "epochs": self.epochs,
                "sg": self.sg,
                "is_gensim": _GENSIM_AVAILABLE and isinstance(self.model, GensimWord2Vec),
            },
            path,
        )
        logger.info("Saved Word2Vec model to %s.", path)

    @classmethod
    def load(cls, filepath: str | Path) -> "Word2VecWrapper":
        """Load fitted Word2Vec model artifact."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No Word2Vec artifact found at {path}")
        data = joblib.load(path)
        instance = cls(
            vector_size=data["vector_size"],
            window=data["window"],
            min_count=data["min_count"],
            epochs=data["epochs"],
            sg=data["sg"],
        )
        instance.model = data["model"]
        instance.is_fitted = data["is_fitted"]
        logger.info("Loaded Word2Vec model from %s.", path)
        return instance
