from __future__ import annotations

import json
import logging
from pathlib import Path

from rag.config import settings
from rag.embeddings import EmbeddingCache, GeminiEmbedder
from rag.ingest.chunker import chunk_documents
from rag.ingest.loaders import load_directory
from rag.models import Chunk, IngestResult
from rag.stores.base import VectorStore
from rag.stores.bm25_store import BM25Store

logger = logging.getLogger(__name__)


def _manifest_path(company_id: str) -> Path:
    return settings.index_dir / "manifest" / f"{company_id}.json"


def _load_manifest(company_id: str) -> dict[str, Chunk]:
    path = _manifest_path(company_id)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {c["chunk_id"]: Chunk.model_validate(c) for c in raw}


def _save_manifest(company_id: str, chunks_by_id: dict[str, Chunk]) -> None:
    path = _manifest_path(company_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump() for c in chunks_by_id.values()]), encoding="utf-8")


class IngestionPipeline:
    """load -> chunk -> embed (cached) -> dense upsert + full BM25 rebuild.

    BM25 has no incremental-update API in bm25s, so every ingestion rebuilds the
    company's sparse index from a persisted manifest of *all* chunks seen so far,
    not just the ones in this batch.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store | None = None,
        embedder: GeminiEmbedder | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.embedder = embedder or GeminiEmbedder()

    def ingest_directory(self, directory: Path, company_id: str) -> IngestResult:
        docs = load_directory(directory, company_id)
        new_chunks = chunk_documents(docs)

        cache = EmbeddingCache(settings.index_dir / "embedding_cache" / f"{company_id}.json")
        texts = [c.text for c in new_chunks]
        embeddings, num_embedded, num_cached = self.embedder.embed(
            texts, task_type="RETRIEVAL_DOCUMENT", cache=cache
        )

        if new_chunks:
            self.vector_store.upsert(company_id, new_chunks, embeddings)

        manifest = _load_manifest(company_id)
        for chunk in new_chunks:
            manifest[chunk.chunk_id] = chunk
        _save_manifest(company_id, manifest)

        all_chunks = list(manifest.values())
        self.bm25_store.build(company_id, all_chunks)

        return IngestResult(
            company_id=company_id,
            documents_loaded=len(docs),
            chunks_created=len(new_chunks),
            chunks_embedded=num_embedded,
            chunks_cached=num_cached,
        )
