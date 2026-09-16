from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.stores.base import DenseMatch
from rag.stores.bm25_store import SparseMatch


def test_rrf_favors_docs_ranked_highly_in_both_lists():
    dense = [DenseMatch("a", 0.9), DenseMatch("b", 0.8), DenseMatch("c", 0.7)]
    sparse = [SparseMatch("b", 5.0), SparseMatch("a", 4.0), SparseMatch("d", 3.0)]

    scores = reciprocal_rank_fusion(dense, sparse, k=60)

    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]
    assert "d" in scores
    assert scores["a"] == scores["b"]  # both appear at ranks {1,2} across the two lists


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([], [], k=60) == {}


def test_rrf_single_list_still_scores():
    dense = [DenseMatch("a", 0.9)]
    scores = reciprocal_rank_fusion(dense, [], k=60)
    assert scores == {"a": 1.0 / 61}
