from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.config import settings

logger = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_BATCH_SIZE = 32


def content_hash(text: str, task_type: TaskType) -> str:
    return hashlib.sha256(f"{task_type}:{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Content-hash -> embedding cache, persisted as one JSON file per company.

    Avoids re-embedding unchanged chunks on repeat ingestion, which matters on a
    free-tier token budget.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def put_many(self, items: dict[str, list[float]]) -> None:
        self._data.update(items)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


class GeminiEmbedder:
    def __init__(self):
        if not settings.use_gemini:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in .env before embedding text."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str], task_type: TaskType) -> list[list[float]]:
        from google.genai import types

        result = self._client.models.embed_content(
            model=settings.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
        return [list(e.values) for e in result.embeddings]

    def embed(
        self,
        texts: list[str],
        task_type: TaskType,
        cache: EmbeddingCache | None = None,
    ) -> tuple[list[list[float]], int, int]:
        """Returns (embeddings, num_embedded_via_api, num_from_cache), in input order."""
        hashes = [content_hash(t, task_type) for t in texts]
        embeddings: list[list[float] | None] = [None] * len(texts)
        to_embed_idx: list[int] = []

        if cache is not None:
            for i, h in enumerate(hashes):
                cached = cache.get(h)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    to_embed_idx.append(i)
        else:
            to_embed_idx = list(range(len(texts)))

        num_cached = len(texts) - len(to_embed_idx)
        newly_embedded: dict[str, list[float]] = {}

        for start in range(0, len(to_embed_idx), _BATCH_SIZE):
            batch_idx = to_embed_idx[start : start + _BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_idx]
            logger.info("Embedding batch of %d texts (task_type=%s)", len(batch_texts), task_type)
            batch_embeddings = self._embed_batch(batch_texts, task_type)
            for idx, emb in zip(batch_idx, batch_embeddings):
                embeddings[idx] = emb
                newly_embedded[hashes[idx]] = emb

        if cache is not None and newly_embedded:
            cache.put_many(newly_embedded)
            cache.save()

        assert all(e is not None for e in embeddings)
        return embeddings, len(to_embed_idx), num_cached  # type: ignore[return-value]
