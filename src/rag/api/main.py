from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from rag.config import settings
from rag.models import IngestResult, QueryResponse
from rag.service import RAGService

app = FastAPI(
    title="RAG Pipeline",
    description="Hybrid (dense + BM25 + cross-encoder rerank) retrieval over company documents.",
)

_service: RAGService | None = None


def get_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service


class QueryRequest(BaseModel):
    query: str
    company_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store": "pinecone" if settings.use_pinecone else "local",
        "gemini_configured": settings.use_gemini,
    }


@app.post("/ingest", response_model=IngestResult)
async def ingest(company_id: str, files: list[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for file in files:
            dest = tmp_dir / file.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        return get_service().ingest(tmp_dir, company_id)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return get_service().query(request.query, request.company_id, mode="answer")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt", response_model=QueryResponse)
def prompt(request: QueryRequest):
    return get_service().query(request.query, request.company_id, mode="prompt")
