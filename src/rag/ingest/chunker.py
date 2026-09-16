from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings
from rag.models import Chunk, RawDocument

_CHARS_PER_TOKEN = 4
"""Rough heuristic (no tokenizer dependency needed for chunk-size budgeting)."""


def _page_for_offset(page_map: list[tuple[int, int]], offset: int) -> int | None:
    if not page_map:
        return None
    page = page_map[0][1]
    for start_offset, page_num in page_map:
        if start_offset > offset:
            break
        page = page_num
    return page


def chunk_document(doc: RawDocument) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens * _CHARS_PER_TOKEN,
        chunk_overlap=settings.chunk_overlap_tokens * _CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    cursor = 0
    for i, text in enumerate(splitter.split_text(doc.text)):
        stripped = text.strip()
        if not stripped:
            continue
        offset = doc.text.find(text, cursor)
        if offset == -1:
            offset = cursor
        cursor = offset + max(len(text) - settings.chunk_overlap_tokens * _CHARS_PER_TOKEN, 1)

        content_hash = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{i}",
                doc_id=doc.doc_id,
                company_id=doc.company_id,
                source_path=doc.source_path,
                title=doc.title,
                text=stripped,
                chunk_index=i,
                page=_page_for_offset(doc.page_map, offset),
                content_hash=content_hash,
            )
        )
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    result: list[Chunk] = []
    for doc in docs:
        result.extend(chunk_document(doc))
    return result
