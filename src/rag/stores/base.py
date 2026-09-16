from __future__ import annotations

from typing import Protocol

from rag.models import Chunk


class DenseMatch:
    def __init__(self, chunk_id: str, score: float):
        self.chunk_id = chunk_id
        self.score = score


class VectorStore(Protocol):
    """Dense vector store. One implementation talks to Pinecone, another is a
    local numpy fallback so the pipeline is fully testable without API keys.
    """

    def upsert(self, company_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def query(
        self, company_id: str, query_embedding: list[float], top_k: int
    ) -> list[DenseMatch]:
        ...

    def get_chunk(self, company_id: str, chunk_id: str) -> Chunk | None:
        ...

    def get_chunks(self, company_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        ...
