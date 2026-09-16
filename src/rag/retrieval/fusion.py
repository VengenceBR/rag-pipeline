from __future__ import annotations

from rag.stores.base import DenseMatch
from rag.stores.bm25_store import SparseMatch


def reciprocal_rank_fusion(
    dense_matches: list[DenseMatch],
    sparse_matches: list[SparseMatch],
    k: int = 60,
) -> dict[str, float]:
    """Standard RRF: score(doc) = sum(1 / (k + rank)) across each ranked list it
    appears in (rank is 1-indexed). Robust to the very different score scales of
    cosine similarity vs. BM25, since only rank position is used.
    """
    scores: dict[str, float] = {}

    for rank, match in enumerate(dense_matches, start=1):
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, match in enumerate(sparse_matches, start=1):
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + rank)

    return scores
