import hashlib

from rag.generation.prompt import build_prompt_package
from rag.ingest.chunker import chunk_documents
from rag.ingest.loaders import load_directory
from rag.models import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.stores.bm25_store import BM25Store
from rag.stores.local_store import LocalVectorStore

_DIM = 64


def _fake_embed(text: str) -> list[float]:
    """Deterministic hashing-vectorizer embedding: no API calls, but words that
    overlap between two texts still pull their vectors closer together, which is
    enough to sanity-check the retrieval pipeline end-to-end.
    """
    vec = [0.0] * _DIM
    for word in text.lower().split():
        h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    return vec


def test_full_pipeline_retrieves_relevant_chunks(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "returns.md").write_text(
        "Refunds are issued within 30 days of a return for unused items.", encoding="utf-8"
    )
    (data_dir / "shipping.md").write_text(
        "Overnight shipping costs thirty four dollars and ninety nine cents.", encoding="utf-8"
    )

    docs = load_directory(data_dir, "acme")
    chunks = chunk_documents(docs)
    assert len(chunks) == 2

    vector_store = LocalVectorStore(base_dir=tmp_path / "vectors")
    bm25_store = BM25Store(base_dir=tmp_path / "bm25")

    embeddings = [_fake_embed(c.text) for c in chunks]
    vector_store.upsert("acme", chunks, embeddings)
    bm25_store.build("acme", chunks)

    query = "how many days do I have to return an item for a refund"
    query_embedding = _fake_embed(query)

    dense = vector_store.query("acme", query_embedding, top_k=5)
    sparse = bm25_store.query("acme", query, top_k=5)
    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    top_chunk_id = max(fused, key=fused.get)
    top_chunk = vector_store.get_chunk("acme", top_chunk_id)
    assert "refund" in top_chunk.text.lower()

    retrieved = [
        RetrievedChunk(chunk=top_chunk, fused_score=fused[top_chunk_id], rerank_score=1.0)
    ]
    package = build_prompt_package(query, retrieved)
    assert "[1]" in package.user_prompt
    assert package.citations[0].chunk_id == top_chunk.chunk_id
