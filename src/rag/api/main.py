from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from rag.auth import ApiKeyRecord, load_api_keys
from rag.config import settings
from rag.models import IngestResult, PublicChatResponse, PublicCitation, QueryResponse
from rag.rate_limit import rate_limiter
from rag.service import RAGService

app = FastAPI(
    title="RAG Pipeline",
    description="Hybrid (dense + BM25 + cross-encoder rerank) retrieval over company documents.",
)

_service: RAGService | None = None
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service


def get_api_key_record(api_key: str | None = Depends(_api_key_header)) -> ApiKeyRecord | None:
    """Resolves X-API-Key to its record, enforcing that key's rate limit along the
    way. Returns None only when auth is disabled (REQUIRE_API_KEY=false), which
    each route treats as "unrestricted" — including no rate limiting.
    """
    if not settings.require_api_key:
        return None
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    record = load_api_keys().get(api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    limit = record.rate_limit_per_minute or settings.rate_limit_per_minute
    allowed, retry_after = rate_limiter.check(api_key, limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({limit} requests/minute for this key).",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return record


def authorize_company(record: ApiKeyRecord | None, company_id: str) -> None:
    if record is None:
        return  # auth disabled
    if not record.authorizes(company_id):
        raise HTTPException(
            status_code=403, detail=f"API key is not authorized for company '{company_id}'."
        )


class QueryRequest(BaseModel):
    query: str
    company_id: str


class PublicChatRequest(BaseModel):
    message: str


def _client_ip(request: Request) -> str:
    # A real deployment behind a reverse proxy (HF Spaces included) puts the
    # original client IP in X-Forwarded-For; fall back to the direct peer.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/", include_in_schema=False)
def index():
    """The public-facing chat site. No API key needed -- it only ever talks to
    settings.public_chat_company_id, through the IP-rate-limited /chat endpoint.
    """
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/chat", response_model=PublicChatResponse)
def public_chat(request: PublicChatRequest, http_request: Request):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > settings.public_chat_max_message_length:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long (max {settings.public_chat_max_message_length} characters).",
        )

    ip = _client_ip(http_request)
    allowed, retry_after = rate_limiter.check(
        f"public:{ip}", settings.public_chat_rate_limit_per_minute
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many messages -- please wait a moment before trying again.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    try:
        result = get_service().query(message, settings.public_chat_company_id, mode="answer")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PublicChatResponse(
        answer=result.answer or "",
        citations=[
            PublicCitation(marker=c.marker, title=c.title, page=c.page) for c in result.citations
        ],
    )


@app.get("/demo", include_in_schema=False)
def demo():
    """A minimal chat page to try the full authenticated API by hand (any company,
    any mode) -- NOT a pattern for real customer-facing use, since it holds the raw
    API key in the browser. A real integration puts the key in the company's own
    backend and proxies from there. The public / + /chat above is the safe pattern.
    """
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store": "pinecone" if settings.use_pinecone else "local",
        "gemini_configured": settings.use_gemini,
        "auth_required": settings.require_api_key,
    }


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    company_id: str,
    files: list[UploadFile],
    record: ApiKeyRecord | None = Depends(get_api_key_record),
):
    authorize_company(record, company_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for file in files:
            safe_name = Path(file.filename or "upload").name
            dest = tmp_dir / safe_name
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        return get_service().ingest(tmp_dir, company_id)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, record: ApiKeyRecord | None = Depends(get_api_key_record)):
    authorize_company(record, request.company_id)
    try:
        return get_service().query(request.query, request.company_id, mode="answer")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt", response_model=QueryResponse)
def prompt(request: QueryRequest, record: ApiKeyRecord | None = Depends(get_api_key_record)):
    authorize_company(record, request.company_id)
    return get_service().query(request.query, request.company_id, mode="prompt")
