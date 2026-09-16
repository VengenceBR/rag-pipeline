from __future__ import annotations

from pathlib import Path

from rag.config import settings
from rag.embeddings import GeminiEmbedder
from rag.generation.generator import GeminiGenerator
from rag.generation.prompt import build_prompt_package
from rag.ingest.pipeline import IngestionPipeline
from rag.models import IngestResult, QueryResponse
from rag.retrieval.rerank import CrossEncoderReranker
from rag.retrieval.retriever import HybridRetriever
from rag.stores.base import VectorStore
from rag.stores.bm25_store import BM25Store
from rag.stores.local_store import LocalVectorStore
from rag.stores.pinecone_store import PineconeVectorStore


def build_vector_store() -> VectorStore:
    if settings.use_pinecone:
        return PineconeVectorStore()
    return LocalVectorStore()


class RAGService:
    """Top-level facade: ingest documents, then query in 'answer' or 'prompt' mode.

    Both CLI and API sit on top of this so behavior can't drift between them.

    Embedder, reranker, and retriever are built lazily on first use: `ingest()` never
    touches the reranker (a ~34MB model download), and neither needs to construct the
    other's dependencies just to run one command.
    """

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or build_vector_store()
        self.bm25_store = BM25Store()
        self._embedder: GeminiEmbedder | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._retriever: HybridRetriever | None = None

    @property
    def embedder(self) -> GeminiEmbedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder()
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_store=self.bm25_store,
                embedder=self.embedder,
                reranker=self.reranker,
            )
        return self._retriever

    def ingest(self, directory: Path, company_id: str, sync: bool = False) -> IngestResult:
        pipeline = IngestionPipeline(
            vector_store=self.vector_store, bm25_store=self.bm25_store, embedder=self.embedder
        )
        return pipeline.ingest_directory(directory, company_id, sync=sync)

    def query(self, query: str, company_id: str, mode: str = "answer") -> QueryResponse:
        retrieved = self.retriever.retrieve(query, company_id)
        prompt_package = build_prompt_package(query, retrieved)

        if mode == "prompt":
            return QueryResponse(
                mode="prompt",
                query=query,
                company_id=company_id,
                prompt_package=prompt_package,
                citations=prompt_package.citations,
                retrieved_chunks=retrieved,
            )

        answer = GeminiGenerator().generate(prompt_package)
        return QueryResponse(
            mode="answer",
            query=query,
            company_id=company_id,
            answer=answer,
            citations=prompt_package.citations,
            retrieved_chunks=retrieved,
        )
