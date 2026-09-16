import hashlib

from rag.config import settings
from rag.ingest.pipeline import IngestionPipeline, _load_manifest
from rag.stores.bm25_store import BM25Store
from rag.stores.local_store import LocalVectorStore

_DIM = 32


class _FakeEmbedder:
    """No API calls: deterministic hashing-vectorizer embedding, same trick as
    test_end_to_end.py. We only care about ids/counts here, not real similarity.
    """

    def embed(self, texts, task_type, cache=None):
        embeddings = []
        for text in texts:
            vec = [0.0] * _DIM
            for word in text.lower().split():
                h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
                vec[h % _DIM] += 1.0
            embeddings.append(vec)
        return embeddings, len(texts), 0


def _make_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "indexes")
    vector_store = LocalVectorStore(base_dir=tmp_path / "vectors")
    bm25_store = BM25Store(base_dir=tmp_path / "bm25")
    pipeline = IngestionPipeline(
        vector_store=vector_store, bm25_store=bm25_store, embedder=_FakeEmbedder()
    )
    return pipeline, vector_store, bm25_store


def test_sync_removes_chunks_for_deleted_file(tmp_path, monkeypatch):
    pipeline, vector_store, _ = _make_pipeline(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.md").write_text("first document about returns", encoding="utf-8")
    (data_dir / "b.md").write_text("second document about shipping", encoding="utf-8")

    result = pipeline.ingest_directory(data_dir, "acme", sync=True)
    assert result.chunks_created == 2
    assert result.chunks_deleted == 0
    assert len(_load_manifest("acme")) == 2

    (data_dir / "b.md").unlink()
    result = pipeline.ingest_directory(data_dir, "acme", sync=True)

    assert result.chunks_deleted == 1
    manifest = _load_manifest("acme")
    assert len(manifest) == 1
    assert "returns" in next(iter(manifest.values())).text
    # actually deleted from the vector store, not just the manifest
    remaining_ids = list(manifest.keys())
    assert vector_store.get_chunks("acme", remaining_ids)
    assert vector_store.query("acme", [0.0] * _DIM, top_k=10).__len__() == 1


def test_no_sync_never_deletes_missing_files(tmp_path, monkeypatch):
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.md").write_text("first document about returns", encoding="utf-8")
    (data_dir / "b.md").write_text("second document about shipping", encoding="utf-8")
    pipeline.ingest_directory(data_dir, "acme", sync=True)

    (data_dir / "b.md").unlink()
    result = pipeline.ingest_directory(data_dir, "acme", sync=False)

    assert result.chunks_deleted == 0
    assert len(_load_manifest("acme")) == 2


def test_edited_file_drops_orphaned_chunks_even_without_sync(tmp_path, monkeypatch):
    """A file that shrinks from 2 chunks to 1 should never leave a stale second
    chunk behind, regardless of sync -- it's still present, just re-chunked.
    """
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    long_text = "Refunds and returns policy details. " * 200
    (data_dir / "a.md").write_text(long_text, encoding="utf-8")

    result = pipeline.ingest_directory(data_dir, "acme", sync=False)
    assert result.chunks_created > 1
    original_chunk_count = result.chunks_created

    (data_dir / "a.md").write_text("Short new content.", encoding="utf-8")
    result = pipeline.ingest_directory(data_dir, "acme", sync=False)

    assert result.chunks_created == 1
    assert result.chunks_deleted == original_chunk_count - 1
    manifest = _load_manifest("acme")
    assert len(manifest) == 1
    assert "Short new content" in next(iter(manifest.values())).text
