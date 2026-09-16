from __future__ import annotations

from rag.config import settings
from rag.models import RetrievedChunk


class CrossEncoderReranker:
    """Local ONNX cross-encoder (FlashRank) — no torch, no API call, ~34MB model
    downloaded and cached on first use.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker

        cache_dir = str(settings.index_dir / "flashrank_cache")
        self._ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": rc.chunk.text} for i, rc in enumerate(candidates)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        reranked: list[RetrievedChunk] = []
        for result in results[:top_k]:
            rc = candidates[result["id"]]
            rc.rerank_score = float(result["score"])
            reranked.append(rc)
        return reranked
