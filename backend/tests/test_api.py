"""
Automated API tests for the TrustRAG FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TrustRAG Backend"
    assert data["documents_indexed"] >= 1
    assert data["chunks_indexed"] >= 1


def test_query_endpoint(client: TestClient):
    payload = {
        "query": "What are the technical skills of the candidate?",
        "top_k": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "citations" in data
    assert len(data["citations"]) > 0
    assert "retrieved_chunks" in data
    assert len(data["retrieved_chunks"]) > 0

    first_citation = data["citations"][0]
    assert "source_file" in first_citation
    assert "page_number" in first_citation
    assert first_citation["page_number"] >= 1


def test_cors_preflight(client: TestClient):
    """
    Verifies that CORS preflight from Vite (http://localhost:5173) is approved.
    """
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/query", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_empty_query_validation(client: TestClient):
    response = client.post("/query", json={"query": "   "})
    assert response.status_code in {400, 422}
