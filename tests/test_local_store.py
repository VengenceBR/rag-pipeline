from rag.models import Chunk
from rag.stores.local_store import LocalVectorStore


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        company_id="acme",
        source_path="x.md",
        title="Doc",
        text=text,
        chunk_index=0,
        content_hash=chunk_id,
    )


def test_upsert_and_query_returns_nearest_by_cosine(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    chunks = [_chunk("a", "returns"), _chunk("b", "shipping"), _chunk("c", "billing")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.upsert("acme", chunks, embeddings)

    matches = store.query("acme", [0.9, 0.1, 0.0], top_k=2)
    assert matches[0].chunk_id == "a"
    assert len(matches) == 2


def test_upsert_overwrites_existing_chunk(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    store.upsert("acme", [_chunk("a", "v1")], [[1.0, 0.0]])
    store.upsert("acme", [_chunk("a", "v2 updated")], [[0.0, 1.0]])

    chunk = store.get_chunk("acme", "a")
    assert chunk.text == "v2 updated"
    matches = store.query("acme", [0.0, 1.0], top_k=5)
    assert len(matches) == 1  # no duplicate row


def test_get_chunks_batch(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    chunks = [_chunk("a", "returns"), _chunk("b", "shipping")]
    store.upsert("acme", chunks, [[1.0, 0.0], [0.0, 1.0]])

    result = store.get_chunks("acme", ["a", "b", "missing"])
    assert set(result.keys()) == {"a", "b"}


def test_query_empty_company_returns_empty(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    assert store.query("nonexistent", [1.0, 0.0], top_k=5) == []
