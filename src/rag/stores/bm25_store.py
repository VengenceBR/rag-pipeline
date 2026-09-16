from __future__ import annotations

import json
from pathlib import Path

import bm25s

from rag.config import settings
from rag.models import Chunk


class SparseMatch:
    def __init__(self, chunk_id: str, score: float):
        self.chunk_id = chunk_id
        self.score = score


class BM25Store:
    """Per-company BM25 lexical index, persisted to disk via bm25s."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (settings.index_dir / "bm25")

    def _company_dir(self, company_id: str) -> Path:
        d = self.base_dir / company_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ids_path(self, company_id: str) -> Path:
        return self._company_dir(company_id) / "chunk_ids.json"

    def build(self, company_id: str, chunks: list[Chunk]) -> None:
        """Rebuilds the full BM25 index for a company from the given chunk set.

        BM25 indexes aren't incrementally updatable in bm25s, so ingestion always
        passes the *complete* current chunk list for the company.
        """
        corpus = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(str(self._company_dir(company_id) / "index"))
        self._ids_path(company_id).write_text(json.dumps(chunk_ids), encoding="utf-8")

    def _load(self, company_id: str) -> tuple[bm25s.BM25, list[str]] | None:
        index_path = self._company_dir(company_id) / "index"
        ids_path = self._ids_path(company_id)
        if not (index_path.exists() and ids_path.exists()):
            return None
        retriever = bm25s.BM25.load(str(index_path), load_corpus=False)
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return retriever, chunk_ids

    def query(self, company_id: str, query_text: str, top_k: int) -> list[SparseMatch]:
        loaded = self._load(company_id)
        if loaded is None:
            return []
        retriever, chunk_ids = loaded
        if not chunk_ids:
            return []
        k = min(top_k, len(chunk_ids))
        query_tokens = bm25s.tokenize(query_text, stopwords="en", show_progress=False)
        results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
        matches = []
        for idx, score in zip(results[0], scores[0]):
            matches.append(SparseMatch(chunk_id=chunk_ids[int(idx)], score=float(score)))
        return matches
