from rag.models import Chunk
from rag.stores.bm25_store import BM25Store


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


def test_build_and_query_ranks_lexical_match_first(tmp_path):
    store = BM25Store(base_dir=tmp_path)
    chunks = [
        _chunk("a", "Refunds are issued within 30 days of a return."),
        _chunk("b", "Overnight shipping costs thirty four dollars."),
        _chunk("c", "Acme Plus membership costs nine ninety nine per month."),
    ]
    store.build("acme", chunks)

    matches = store.query("acme", "how do refunds and returns work", top_k=2)
    assert matches[0].chunk_id == "a"


def test_query_before_build_returns_empty(tmp_path):
    store = BM25Store(base_dir=tmp_path)
    assert store.query("nonexistent", "anything", top_k=5) == []


def test_rebuild_replaces_previous_index(tmp_path):
    store = BM25Store(base_dir=tmp_path)
    store.build("acme", [_chunk("a", "shipping information")])
    store.build("acme", [_chunk("b", "billing information")])

    matches = store.query("acme", "information", top_k=5)
    ids = {m.chunk_id for m in matches}
    assert ids == {"b"}


def test_build_with_empty_corpus_does_not_raise(tmp_path):
    """Regression: bm25s.index() raises ValueError on an empty corpus. Hit by the
    watcher's initial sync against an empty company directory.
    """
    store = BM25Store(base_dir=tmp_path)
    store.build("acme", [])
    assert store.query("acme", "anything", top_k=5) == []


def test_build_empty_after_nonempty_clears_stale_index(tmp_path):
    """Regression: deleting a company's last doc (sync ingest -> 0 chunks) must not
    leave the previous index around to serve stale query results.
    """
    store = BM25Store(base_dir=tmp_path)
    store.build("acme", [_chunk("a", "shipping information")])
    assert store.query("acme", "information", top_k=5) != []

    store.build("acme", [])
    assert store.query("acme", "information", top_k=5) == []
