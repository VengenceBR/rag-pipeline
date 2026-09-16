from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.config import settings
from rag.models import Chunk
from rag.stores.base import DenseMatch


class LocalVectorStore:
    """Numpy cosine-similarity store. Zero external dependencies, zero API keys.

    Lets the full retrieval pipeline run and be tested before Pinecone is provisioned,
    and works as a lightweight self-hosted fallback if you never want Pinecone at all.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (settings.index_dir / "local")

    def _company_dir(self, company_id: str) -> Path:
        d = self.base_dir / company_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self, company_id: str) -> tuple[np.ndarray, list[Chunk], list[str]]:
        d = self._company_dir(company_id)
        vec_path = d / "vectors.npy"
        meta_path = d / "chunks.json"
        if not vec_path.exists() or not meta_path.exists():
            return np.zeros((0, 0), dtype=np.float32), [], []
        vectors = np.load(vec_path)
        raw_chunks = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(c) for c in raw_chunks]
        ids = [c.chunk_id for c in chunks]
        return vectors, chunks, ids

    def _save(self, company_id: str, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        d = self._company_dir(company_id)
        np.save(d / "vectors.npy", vectors)
        (d / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in chunks]), encoding="utf-8"
        )

    def upsert(self, company_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        existing_vectors, existing_chunks, existing_ids = self._load(company_id)
        id_to_row = {cid: i for i, cid in enumerate(existing_ids)}

        new_vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vectors = new_vectors / norms

        if existing_vectors.shape[0] == 0:
            existing_vectors = np.zeros((0, new_vectors.shape[1]), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in id_to_row:
                row = id_to_row[chunk.chunk_id]
                existing_vectors[row] = new_vectors[i]
                existing_chunks[row] = chunk
            else:
                existing_vectors = np.vstack([existing_vectors, new_vectors[i : i + 1]])
                existing_chunks.append(chunk)
                id_to_row[chunk.chunk_id] = len(existing_chunks) - 1

        self._save(company_id, existing_vectors, existing_chunks)

    def query(
        self, company_id: str, query_embedding: list[float], top_k: int
    ) -> list[DenseMatch]:
        vectors, chunks, ids = self._load(company_id)
        if len(chunks) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        scores = vectors @ q
        top_idx = np.argsort(-scores)[:top_k]
        return [DenseMatch(chunk_id=ids[i], score=float(scores[i])) for i in top_idx]

    def get_chunk(self, company_id: str, chunk_id: str) -> Chunk | None:
        _, chunks, ids = self._load(company_id)
        for chunk, cid in zip(chunks, ids):
            if cid == chunk_id:
                return chunk
        return None

    def get_chunks(self, company_id: str, chunk_ids: list[str]) -> dict[str, Chunk]:
        wanted = set(chunk_ids)
        _, chunks, ids = self._load(company_id)
        return {cid: chunk for chunk, cid in zip(chunks, ids) if cid in wanted}
