"""Integration tests for the ReviewIQ FastAPI backend service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "service" in data


class TestSearchEndpoints:
    def test_search_query(self, client: TestClient) -> None:
        payload = {"query": "laptop with long battery life", "top_k": 3, "rerank": False}
        response = client.post("/search/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "laptop with long battery life"
        assert "results" in data
        assert len(data["results"]) <= 3
        if data["results"]:
            first = data["results"][0]
            assert "rank" in first
            assert "score" in first
            assert "text" in first

    def test_search_ner(self, client: TestClient) -> None:
        payload = {"text": "Dell XPS 15 with 16GB RAM and RTX 4060 GPU and 15.6 inch display"}
        response = client.post("/search/ner", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "grouped" in data
        labels = [e["label"] for e in data["entities"]]
        assert "BRAND" in labels or "GPU" in labels or "RAM" in labels


class TestSentimentEndpoints:
    def test_sentiment_classify(self, client: TestClient) -> None:
        payload = {"text": "This laptop is amazing! Battery lasts all day and performance is super fast."}
        response = client.post("/sentiment/classify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["score"] > 0.05
        assert data["confidence"] >= 0.5

    def test_sentiment_aspects(self, client: TestClient) -> None:
        payload = {"text": "Great battery life, but terrible thermal cooling and loud fans."}
        response = client.post("/sentiment/aspects", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "extracted_aspects" in data
        assert "scored_aspects" in data
        assert "battery" in data["scored_aspects"]
        assert "thermal" in data["scored_aspects"]
        assert data["scored_aspects"]["battery"]["sentiment"] == "positive"
        assert data["scored_aspects"]["thermal"]["sentiment"] == "negative"


class TestRecommendationEndpoints:
    def test_recommend_rank_valid(self, client: TestClient) -> None:
        payload = {
            "user_weights": {"battery": 2.5, "display": 1.5, "thermal": 0.5},
            "top_n": 3,
        }
        response = client.post("/recommend/rank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 3
        top_rec = data["recommendations"][0]
        assert top_rec["rank"] == 1
        assert "composite_score" in top_rec
        assert "explanation" in top_rec

    def test_recommend_rank_negative_weight_bad_request(self, client: TestClient) -> None:
        payload = {
            "user_weights": {"battery": -1.0},
            "top_n": 3,
        }
        response = client.post("/recommend/rank", json=payload)
        assert response.status_code == 400


class TestSummarizationEndpoints:
    def test_summary_textrank(self, client: TestClient) -> None:
        review_text = (
            "The battery life is exceptional and easily lasts through a full workday. "
            "The screen is bright and vivid with great contrast. "
            "However, the fans are noisy and the chassis gets hot during gaming sessions. "
            "Customer service was prompt and responsive."
        )
        payload = {"text": review_text, "num_sentences": 2}
        response = client.post("/summary/textrank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["sentence_count"] <= 2
        assert len(data["summary"]) > 0


class TestRAGEndpoints:
    def test_rag_query(self, client: TestClient) -> None:
        payload = {"query": "How is the battery life on these laptops?", "top_k": 3}
        response = client.post("/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert len(data["citations"]) <= 3
