import pytest
from fastapi.testclient import TestClient

from rag.config import settings
from rag.models import Citation, QueryResponse


class _FakeService:
    def query(self, query, company_id, mode="answer"):
        assert company_id == settings.public_chat_company_id  # always locked to this
        return QueryResponse(
            mode=mode,
            query=query,
            company_id=company_id,
            answer=f"answer to: {query}",
            citations=[
                Citation(
                    marker="[1]",
                    source_path="data/corvit/secret_internal_path.md",
                    title="Corvit Policy",
                    page=2,
                    chunk_id="internal-chunk-id",
                )
            ],
        )


class _FailingService:
    def query(self, query, company_id, mode="answer"):
        raise RuntimeError("GEMINI_API_KEY is not set.")


@pytest.fixture
def client(monkeypatch):
    import rag.api.main as api_main

    monkeypatch.setattr(api_main, "get_service", lambda: _FakeService())
    return TestClient(api_main.app)


def _headers(ip: str) -> dict:
    return {"X-Forwarded-For": ip}  # gives each test its own rate-limit bucket


def test_chat_returns_answer_and_trimmed_citations(client):
    r = client.post("/chat", json={"message": "hi"}, headers=_headers("1.1.1.1"))
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "answer to: hi"
    assert body["citations"] == [{"marker": "[1]", "title": "Corvit Policy", "page": 2}]
    # internal details must never reach the public response
    assert "secret_internal_path" not in r.text
    assert "internal-chunk-id" not in r.text


def test_chat_rejects_empty_message(client):
    r = client.post("/chat", json={"message": "   "}, headers=_headers("1.1.1.2"))
    assert r.status_code == 400


def test_chat_rejects_overlong_message(client, monkeypatch):
    monkeypatch.setattr(settings, "public_chat_max_message_length", 10)
    r = client.post(
        "/chat", json={"message": "x" * 11}, headers=_headers("1.1.1.3")
    )
    assert r.status_code == 400


def test_chat_never_requires_an_api_key(client):
    r = client.post("/chat", json={"message": "no key here"}, headers=_headers("1.1.1.4"))
    assert r.status_code == 200


def test_chat_is_rate_limited_per_ip(client, monkeypatch):
    monkeypatch.setattr(settings, "public_chat_rate_limit_per_minute", 1)
    headers = _headers("1.1.1.5")

    first = client.post("/chat", json={"message": "one"}, headers=headers)
    assert first.status_code == 200

    second = client.post("/chat", json={"message": "two"}, headers=headers)
    assert second.status_code == 429
    assert "Retry-After" in second.headers

    # a different IP has its own independent budget
    third = client.post("/chat", json={"message": "three"}, headers=_headers("1.1.1.6"))
    assert third.status_code == 200


def test_chat_maps_service_runtime_error_to_503(monkeypatch):
    import rag.api.main as api_main

    monkeypatch.setattr(api_main, "get_service", lambda: _FailingService())
    client = TestClient(api_main.app)

    r = client.post("/chat", json={"message": "hi"}, headers=_headers("1.1.1.7"))
    assert r.status_code == 503


def test_index_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
