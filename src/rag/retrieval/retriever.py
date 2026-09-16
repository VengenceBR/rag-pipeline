from __future__ import annotations

from rag.config import settings
from rag.embeddings import GeminiEmbedder
from rag.models import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.rerank import CrossEncoderReranker
from rag.stores.base import VectorStore
from rag.stores.bm25_store import BM25Store


class HybridRetriever:
    """dense (Pinecone/local) + sparse (BM25) -> RRF fusion -> cross-encoder rerank."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: GeminiEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder or GeminiEmbedder()
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str, company_id: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.rerank_top_k

        [query_embedding], _, _ = self.embedder.embed([query], task_type="RETRIEVAL_QUERY")

        dense_matches = self.vector_store.query(company_id, query_embedding, settings.dense_top_k)
        sparse_matches = self.bm25_store.query(company_id, query, settings.sparse_top_k)

        fused_scores = reciprocal_rank_fusion(dense_matches, sparse_matches, k=settings.rrf_k)
        if not fused_scores:
            return []

        chunk_ids = list(fused_scores.keys())
        chunks_by_id = self.vector_store.get_chunks(company_id, chunk_ids)

        dense_score_by_id = {m.chunk_id: m.score for m in dense_matches}
        sparse_score_by_id = {m.chunk_id: m.score for m in sparse_matches}

        candidates: list[RetrievedChunk] = []
        for chunk_id, fused_score in sorted(fused_scores.items(), key=lambda kv: -kv[1]):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_score_by_id.get(chunk_id),
                    sparse_score=sparse_score_by_id.get(chunk_id),
                    fused_score=fused_score,
                )
            )

        return self.reranker.rerank(query, candidates, top_k=top_k)
