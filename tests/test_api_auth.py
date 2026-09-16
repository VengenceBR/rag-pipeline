import pytest
from fastapi.testclient import TestClient

from rag.auth import create_api_key
from rag.config import settings
from rag.models import IngestResult, QueryResponse


class _FakeService:
    """Stands in for RAGService so these tests exercise only the auth boundary,
    never a real Gemini/Pinecone call.
    """

    def ingest(self, directory, company_id):
        return IngestResult(
            company_id=company_id, documents_loaded=0, chunks_created=0,
            chunks_embedded=0, chunks_cached=0,
        )

    def query(self, query, company_id, mode="answer"):
        from rag.models import Citation

        return QueryResponse(
            mode=mode,
            query=query,
            company_id=company_id,
            answer="ok",
            citations=[
                Citation(
                    marker="[1]", source_path="data/x.md", title="x", page=None, chunk_id="c1"
                )
            ],
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    import rag.api.main as api_main

    monkeypatch.setattr(settings, "api_keys_file", tmp_path / "api_keys.json")
    monkeypatch.setattr(settings, "require_api_key", True)
    monkeypatch.setattr(api_main, "get_service", lambda: _FakeService())
    return TestClient(api_main.app)


def test_query_without_key_is_401(client):
    r = client.post("/query", json={"query": "hi", "company_id": "acme"})
    assert r.status_code == 401


def test_query_with_invalid_key_is_401(client):
    r = client.post(
        "/query",
        json={"query": "hi", "company_id": "acme"},
        headers={"X-API-Key": "sk-not-a-real-key"},
    )
    assert r.status_code == 401


def test_query_with_key_for_other_company_is_403(client):
    key = create_api_key(name="acme bot", companies=["acme"])
    r = client.post(
        "/query",
        json={"query": "hi", "company_id": "someone-else"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 403


def test_query_with_authorized_key_succeeds(client):
    key = create_api_key(name="acme bot", companies=["acme"])
    r = client.post(
        "/query",
        json={"query": "hi", "company_id": "acme"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "ok"


def test_wildcard_key_authorizes_any_company(client):
    key = create_api_key(name="admin", companies=["*"])
    r = client.post(
        "/query",
        json={"query": "hi", "company_id": "any-company"},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200


def test_auth_disabled_allows_requests_without_key(client, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)
    r = client.post("/query", json={"query": "hi", "company_id": "acme"})
    assert r.status_code == 200


def test_health_never_requires_a_key(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_exceeding_rate_limit_returns_429_with_retry_after(client):
    key = create_api_key(name="acme bot", companies=["acme"], rate_limit_per_minute=1)
    headers = {"X-API-Key": key}
    body = {"query": "hi", "company_id": "acme"}

    first = client.post("/query", json=body, headers=headers)
    assert first.status_code == 200

    second = client.post("/query", json=body, headers=headers)
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_rate_limit_is_scoped_per_key(client):
    key_a = create_api_key(name="a", companies=["acme"], rate_limit_per_minute=1)
    key_b = create_api_key(name="b", companies=["acme"], rate_limit_per_minute=1)
    body = {"query": "hi", "company_id": "acme"}

    assert client.post("/query", json=body, headers={"X-API-Key": key_a}).status_code == 200
    # key_a is now over budget, but key_b has its own independent window
    assert client.post("/query", json=body, headers={"X-API-Key": key_b}).status_code == 200
    assert client.post("/query", json=body, headers={"X-API-Key": key_a}).status_code == 429
