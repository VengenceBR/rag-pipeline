from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """A single loaded source document, before chunking."""

    doc_id: str
    company_id: str
    source_path: str
    title: str
    text: str
    page_map: list[tuple[int, int]] = Field(default_factory=list)
    """List of (char_offset, page_number) breakpoints, for PDFs. Empty for flat text sources."""


class Chunk(BaseModel):
    """A chunk of a document, ready to embed and index."""

    chunk_id: str
    doc_id: str
    company_id: str
    source_path: str
    title: str
    text: str
    chunk_index: int
    page: int | None = None
    content_hash: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: str
    """e.g. '[1]'"""
    source_path: str
    title: str
    page: int | None = None
    chunk_id: str


class PromptPackage(BaseModel):
    """A fully-assembled prompt, handed back without generation (mode='prompt')."""

    system_prompt: str
    user_prompt: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class QueryResponse(BaseModel):
    mode: Literal["answer", "prompt"]
    query: str
    company_id: str
    answer: str | None = None
    prompt_package: PromptPackage | None = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class IngestResult(BaseModel):
    company_id: str
    documents_loaded: int
    chunks_created: int
    chunks_embedded: int
    chunks_cached: int
