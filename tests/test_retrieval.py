"""Unit and integration tests for Phase 4 (Information Retrieval & Laptop NER).

Covers:
- TFIDFRetriever (sparse lexical retrieval, cosine scoring, top-k ranking, serialization)
- FAISSRetriever (dense semantic retrieval, FAISS inner-product search, index persistence)
- Reciprocal Rank Fusion (RRF score calculation and rank ordering)
- HybridRetriever (multi-stage candidate gathering, fusion, neural reranking fallback, persistence)
- LaptopNER (hardware spec extraction across Brand, CPU, GPU, RAM, Storage, Display, Battery, OS, tagging)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from reviewiq.retrieval.hybrid_search import HybridRetriever, reciprocal_rank_fusion
from reviewiq.retrieval.ner import LaptopNER
from reviewiq.retrieval.semantic_search import FAISSRetriever
from reviewiq.retrieval.tfidf_retrieval import SearchResult, TFIDFRetriever


@pytest.fixture
def sample_laptop_corpus() -> list[str]:
    return [
        "Apple MacBook Pro with M3 Pro chip, 16GB unified memory and 512GB SSD. Incredible battery lasts 16 hours.",
        "Dell XPS 15 with Intel Core i7-13700H, 32GB RAM, 1TB SSD, and NVIDIA RTX 4060 GPU. OLED 3.5K display is gorgeous.",
        "Lenovo ThinkPad X1 Carbon with Intel Core i7, lightweight carbon fiber build, great keyboard for typing.",
        "ASUS ROG Zephyrus G14 with AMD Ryzen 9 7940HS and RTX 4070, 14-inch 165Hz gaming screen.",
        "Budget Acer Aspire 5 with Intel Core i3, 8GB RAM, and 256GB SSD. Good value for student work.",
        "HP Envy x360 2-in-1 touchscreen convertible laptop with AMD Ryzen 5, 16GB RAM, decent battery backup.",
    ]


@pytest.fixture
def sample_corpus_ids(sample_laptop_corpus) -> list[str]:
    return [f"DOC_{i}" for i in range(len(sample_laptop_corpus))]


# ==============================================================================
# TF-IDF Retrieval Tests
# ==============================================================================

class TestTFIDFRetriever:
    def test_index_and_search_exact_match(self, sample_laptop_corpus, sample_corpus_ids):
        retriever = TFIDFRetriever(max_features=500)
        retriever.index(sample_laptop_corpus, doc_ids=sample_corpus_ids)

        assert retriever.is_indexed
        assert retriever.doc_matrix is not None
        assert retriever.doc_matrix.shape[0] == len(sample_laptop_corpus)

        results = retriever.search("Apple MacBook M3 battery", top_k=2)
        assert len(results) > 0
        assert results[0].doc_id == "DOC_0"  # Apple review is rank 1
        assert "Apple" in results[0].text
        assert results[0].score > 0.0

    def test_search_out_of_vocab_or_empty(self, sample_laptop_corpus):
        retriever = TFIDFRetriever()
        retriever.index(sample_laptop_corpus)

        assert retriever.search("") == []
        assert retriever.search("   ") == []
        assert retriever.search("xyzzyunknownterm") == []

    def test_search_score_threshold(self, sample_laptop_corpus):
        retriever = TFIDFRetriever()
        retriever.index(sample_laptop_corpus)

        # High threshold should filter out low matches
        results = retriever.search("gaming graphics", top_k=5, min_score=0.99)
        assert len(results) == 0

    def test_save_load(self, sample_laptop_corpus, sample_corpus_ids):
        retriever = TFIDFRetriever()
        retriever.index(sample_laptop_corpus, doc_ids=sample_corpus_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tfidf_index.joblib"
            retriever.save(save_path)
            assert save_path.exists()

            loaded = TFIDFRetriever.load(save_path)
            assert loaded.is_indexed
            res = loaded.search("ThinkPad keyboard", top_k=1)
            assert len(res) == 1
            assert res[0].doc_id == "DOC_2"


# ==============================================================================
# FAISS Semantic Search Tests
# ==============================================================================

class TestFAISSRetriever:
    def test_index_and_search_mock_embeddings(self, sample_laptop_corpus, sample_corpus_ids):
        # Generate synthetic normalized embeddings for fast, deterministic unit test
        np.random.seed(42)
        dim = 16
        mock_embeddings = np.random.randn(len(sample_laptop_corpus), dim).astype(np.float32)

        retriever = FAISSRetriever()
        retriever.index(
            sample_laptop_corpus,
            doc_ids=sample_corpus_ids,
            embeddings=mock_embeddings,
        )

        assert retriever.is_indexed
        assert retriever.index_ is not None
        assert retriever.index_.ntotal == len(sample_laptop_corpus)

        # Monkey-patch encode for search to return first doc's vector
        retriever.encode = lambda texts: mock_embeddings[0:1]

        results = retriever.search("sample query", top_k=3)
        assert len(results) == 3
        # First result should match mock_embeddings[0] with cosine sim ~ 1.0
        assert results[0].doc_id == "DOC_0"
        assert pytest.approx(results[0].score, abs=1e-4) == 1.0

    def test_save_load_faiss(self, sample_laptop_corpus, sample_corpus_ids):
        np.random.seed(42)
        dim = 16
        mock_embeddings = np.random.randn(len(sample_laptop_corpus), dim).astype(np.float32)

        retriever = FAISSRetriever()
        retriever.index(
            sample_laptop_corpus,
            doc_ids=sample_corpus_ids,
            embeddings=mock_embeddings,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "faiss.index"
            meta_path = Path(tmpdir) / "faiss.meta.joblib"

            retriever.save(idx_path, meta_path)
            assert idx_path.exists()
            assert meta_path.exists()

            loaded = FAISSRetriever.load(idx_path, meta_path)
            assert loaded.is_indexed
            assert loaded.index_.ntotal == len(sample_laptop_corpus)
            assert loaded.documents == sample_laptop_corpus


# ==============================================================================
# Reciprocal Rank Fusion & Hybrid Search Tests
# ==============================================================================

class TestHybridRetrieval:
    def test_reciprocal_rank_fusion(self):
        list1 = [
            SearchResult(doc_id="A", text="Doc A", score=0.9, rank=1),
            SearchResult(doc_id="B", text="Doc B", score=0.8, rank=2),
            SearchResult(doc_id="C", text="Doc C", score=0.7, rank=3),
        ]
        list2 = [
            SearchResult(doc_id="B", text="Doc B", score=0.95, rank=1),
            SearchResult(doc_id="A", text="Doc A", score=0.85, rank=2),
            SearchResult(doc_id="D", text="Doc D", score=0.6, rank=3),
        ]

        fused = reciprocal_rank_fusion([list1, list2], rrf_k=60, top_k=4)

        assert len(fused) == 4
        # Both A and B are in positions 1 and 2 in list1 and list2
        # Score for A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        # Score for B: 1/(60+2) + 1/(60+1) = 0.032522
        # They should be top 2
        top_ids = [fused[0].doc_id, fused[1].doc_id]
        assert "A" in top_ids
        assert "B" in top_ids
        assert fused[0].rank == 1
        assert fused[1].rank == 2

    def test_hybrid_retriever_search_without_reranker(self, sample_laptop_corpus, sample_corpus_ids):
        sparse = TFIDFRetriever(max_features=200)
        dense = FAISSRetriever()

        # Provide synthetic embeddings for dense to avoid slow online model downloads during testing
        np.random.seed(123)
        mock_emb = np.random.randn(len(sample_laptop_corpus), 16).astype(np.float32)

        sparse.index(sample_laptop_corpus, doc_ids=sample_corpus_ids)
        dense.index(sample_laptop_corpus, doc_ids=sample_corpus_ids, embeddings=mock_emb)
        dense.encode = lambda texts: mock_emb[1:2]  # Query matches DOC_1

        hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=60)
        hybrid.is_indexed = True

        results = hybrid.search("Intel RTX gaming", candidate_k=4, final_k=2, rerank=False)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_save_load_hybrid(self, sample_laptop_corpus, sample_corpus_ids):
        sparse = TFIDFRetriever(max_features=200)
        dense = FAISSRetriever()
        mock_emb = np.random.randn(len(sample_laptop_corpus), 16).astype(np.float32)

        sparse.index(sample_laptop_corpus, doc_ids=sample_corpus_ids)
        dense.index(sample_laptop_corpus, doc_ids=sample_corpus_ids, embeddings=mock_emb)

        hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)
        hybrid.is_indexed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "hybrid_retriever"
            hybrid.save(save_dir)
            assert (save_dir / "sparse_tfidf.joblib").exists()
            assert (save_dir / "dense_faiss.index").exists()

            loaded = HybridRetriever.load(save_dir)
            assert loaded.is_indexed
            assert loaded.sparse_retriever.is_indexed
            assert loaded.dense_retriever.is_indexed


# ==============================================================================
# Laptop NER Tests
# ==============================================================================

class TestLaptopNER:
    @pytest.fixture
    def ner(self) -> LaptopNER:
        return LaptopNER()

    def test_brand_extraction(self, ner):
        text = "I compared Dell, Apple MacBook, Lenovo Legion and ASUS ROG."
        grouped = ner.extract_entities_grouped(text)

        assert "BRAND" in grouped
        brands = set(grouped["BRAND"])
        assert "Dell" in brands
        assert "Apple" in brands
        assert "Lenovo" in brands
        assert "ASUS" in brands

    def test_cpu_extraction(self, ner):
        text = "Powered by Intel Core i7-13700H and AMD Ryzen 7 7840HS, or Apple M3 Pro."
        grouped = ner.extract_entities_grouped(text)

        assert "CPU" in grouped
        cpus = " ".join(grouped["CPU"])
        assert "i7" in cpus or "Core" in cpus
        assert "Ryzen" in cpus
        assert "M3" in cpus

    def test_gpu_extraction(self, ner):
        text = "Equipped with NVIDIA RTX 4060 and Intel Iris Xe graphics."
        grouped = ner.extract_entities_grouped(text)

        assert "GPU" in grouped
        gpus = " ".join(grouped["GPU"])
        assert "4060" in gpus or "RTX" in gpus
        assert "Iris" in gpus or "Xe" in gpus

    def test_ram_and_storage_extraction(self, ner):
        text = "Upgraded to 32GB RAM and 1TB SSD for fast loading."
        grouped = ner.extract_entities_grouped(text)

        assert "RAM" in grouped
        assert "STORAGE" in grouped
        assert any("32gb" in r.lower() for r in grouped["RAM"])
        assert any("1tb" in s.lower() for s in grouped["STORAGE"])

    def test_display_and_battery(self, ner):
        text = "The 14-inch 120Hz OLED screen looks great, and the 70Wh battery lasts 10 hours of battery."
        grouped = ner.extract_entities_grouped(text)

        assert "DISPLAY" in grouped
        displays = " ".join(grouped["DISPLAY"]).lower()
        assert "oled" in displays or "120hz" in displays or "14" in displays
        assert "BATTERY" in grouped

    def test_tag_text(self, ner):
        text = "Great Dell laptop with 16GB RAM."
        tagged = ner.tag_text(text)

        assert "[Dell](BRAND)" in tagged
        assert "[16GB](RAM)" in tagged or "[16GB RAM](RAM)" in tagged
