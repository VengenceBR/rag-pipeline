from __future__ import annotations

from rag.config import settings
from rag.models import Chunk
from rag.stores.base import DenseMatch

_METADATA_FIELDS = ("doc_id", "company_id", "source_path", "title", "chunk_index", "page", "text")


class PineconeVectorStore:
    """Dense store backed by Pinecone serverless.

    Namespaced per company_id so tenants never share a search space, on a single
    shared index (cheaper on the free tier than one index per company).
    """

    def __init__(self):
        if not settings.use_pinecone:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._ensure_index()
        self._index = self._pc.Index(name=settings.pinecone_index_name)

    def _ensure_index(self) -> None:
        existing = {idx["name"] for idx in self._pc.list_indexes()}
        if settings.pinecone_index_name in existing:
            return
        from pinecone import ServerlessSpec

        self._pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    def upsert(self, company_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {field: getattr(chunk, field) for field in _METADATA_FIELDS}
            metadata = {k: v for k, v in metadata.items() if v is not None}
            vectors.append({"id": chunk.chunk_id, "values": embedding, "metadata": metadata})

        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[start : start + batch_size], namespace=company_id)

    def query(
        self, company_id: str, query_embedding: list[float], top_k: int
    ) -> list[DenseMatch]:
        response = self._index.query(
            namespace=company_id,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=False,
        )
        return [DenseMatch(chunk_id=m["id"], score=float(m["score"])) for m in response["matches"]]

    def get_chunk(self, company_id: str, chunk_id: str) -> Chunk | None:
        return self.get_chunks(company_id, [chunk_id]).get(chunk_id)

    def get_chunks(self, company_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        result: dict[str, Chunk] = {}
        batch_size = 100
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            response = self._index.fetch(ids=batch, namespace=company_id)
            for chunk_id, record in response.get("vectors", {}).items():
                metadata = record["metadata"]
                result[chunk_id] = Chunk(
                    chunk_id=chunk_id,
                    doc_id=metadata["doc_id"],
                    company_id=metadata["company_id"],
                    source_path=metadata["source_path"],
                    title=metadata["title"],
                    text=metadata.get("text", ""),
                    chunk_index=metadata["chunk_index"],
                    page=metadata.get("page"),
                    content_hash="",
                )
        return result

    def delete(self, company_id: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        batch_size = 1000
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start : start + batch_size]
            self._index.delete(ids=batch, namespace=company_id)
