from rag.ingest.chunker import chunk_document
from rag.models import RawDocument


def test_chunk_document_splits_long_text():
    text = ("This is a sentence about returns and refunds. " * 100).strip()
    doc = RawDocument(
        doc_id="d1", company_id="acme", source_path="x.md", title="Doc", text=text
    )
    chunks = chunk_document(doc)

    assert len(chunks) > 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.company_id == "acme" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_small_text_single_chunk():
    doc = RawDocument(
        doc_id="d2", company_id="acme", source_path="y.md", title="Doc", text="Short text."
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunk_document_tracks_page_numbers():
    text = "Page one content. " * 50 + "Page two content. " * 50
    page_map = [(0, 1), (len("Page one content. " * 50), 2)]
    doc = RawDocument(
        doc_id="d3",
        company_id="acme",
        source_path="z.pdf",
        title="Doc",
        text=text,
        page_map=page_map,
    )
    chunks = chunk_document(doc)
    pages = {c.page for c in chunks}
    assert pages <= {1, 2}
    assert 1 in pages
