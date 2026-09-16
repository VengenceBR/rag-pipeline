from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from rag.auth import ApiKeyRecord, load_api_keys
from rag.config import settings
from rag.models import IngestResult, QueryResponse
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
    """Resolves X-API-Key to its record. Returns None only when auth is disabled
    (REQUIRE_API_KEY=false), which each route treats as "unrestricted".
    """
    if not settings.require_api_key:
        return None
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    record = load_api_keys().get(api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")
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
