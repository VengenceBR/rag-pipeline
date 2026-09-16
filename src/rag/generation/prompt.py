from __future__ import annotations

from rag.models import Citation, PromptPackage, RetrievedChunk

SYSTEM_PROMPT = """You are a support assistant for this company. Answer the user's question \
using ONLY the numbered context blocks below. Every claim must cite the block(s) it came from \
using its marker, e.g. [1] or [1][3]. If the context does not contain the answer, say so plainly \
instead of guessing."""


def build_prompt_package(query: str, retrieved: list[RetrievedChunk]) -> PromptPackage:
    context_blocks = []
    citations: list[Citation] = []

    for i, rc in enumerate(retrieved, start=1):
        marker = f"[{i}]"
        chunk = rc.chunk
        location = f"{chunk.title}" + (f", p.{chunk.page}" if chunk.page else "")
        context_blocks.append(f"{marker} (source: {location})\n{chunk.text}")
        citations.append(
            Citation(
                marker=marker,
                source_path=chunk.source_path,
                title=chunk.title,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
            )
        )

    context_str = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    user_prompt = f"Context:\n\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    return PromptPackage(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        citations=citations,
        retrieved_chunks=retrieved,
    )
