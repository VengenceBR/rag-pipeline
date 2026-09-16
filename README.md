# rag-pipeline

End-to-end hybrid RAG pipeline for a company's documents: dense (Pinecone) + BM25 lexical
retrieval, fused with Reciprocal Rank Fusion, reordered by a local cross-encoder reranker.
Generation and embeddings run on the Gemini free tier. Multi-tenant (`company_id`) from the
ground up.

```
ingest:  load (pdf/docx/html/md/txt) → chunk → embed (Gemini) → Pinecone upsert + BM25 index
query:   embed query → dense top-25 + sparse top-25 → RRF fuse → cross-encoder rerank top-8
              → mode=answer  → Gemini grounded, cited answer
                mode=prompt  → the assembled prompt handed back, ungenerated
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
copy .env.example .env        # then fill in GEMINI_API_KEY (required)
```

`PINECONE_API_KEY` is optional — leave it unset and the pipeline runs on a local numpy
vector store instead (same interface, zero external dependency, good for dev/demo).

## Try it (sample data included)

```bash
python -m rag.cli ingest data/sample_company --company acme
python -m rag.cli query "how many days do I have to return an item?" --company acme
python -m rag.cli prompt "what's the restocking fee on large appliances?" --company acme
python -m rag.cli search "lost package" --company acme --top-k 5
python -m rag.cli stats --company acme
python -m rag.cli check-models
```

## Run the API

```bash
uvicorn rag.api.main:app --reload
```

Every route except `/health` requires an `X-API-Key` header, scoped to specific `company_id`s
so one client can never query or ingest into another company's index. Manage keys with the CLI:

```bash
python -m rag.cli keys create --name "acme support bot" --company acme
python -m rag.cli keys list
python -m rag.cli keys revoke sk-...
```

Set `REQUIRE_API_KEY=false` in `.env` to disable auth entirely (local dev only).

- `GET /health` — no key required
- `POST /ingest?company_id=acme` (multipart file upload)
- `POST /query` `{"query": "...", "company_id": "acme"}` → grounded answer + citations
- `POST /prompt` `{"query": "...", "company_id": "acme"}` → assembled prompt package, no generation

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: sk-..." -H "Content-Type: application/json" \
  -d '{"query": "how many days do I have to return an item?", "company_id": "acme"}'
```

## Tests

```bash
pytest
```

The full suite runs with no API keys set — it exercises chunking, RRF fusion, the local
vector store, BM25, and API auth (401/403/200 paths) directly, plus an end-to-end retrieval
pass with a fake embedder.

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Config

All tuning knobs live in `.env` (see `.env.example`): chunk size/overlap, dense/sparse
top-k, RRF `k`, rerank top-k, and the Gemini model IDs. `python -m rag.cli check-models`
confirms your API key can actually reach the configured model IDs before you rely on them.

## Design notes

- **Hybrid retrieval**: dense embeddings catch paraphrase, BM25 catches exact tokens (SKUs,
  error codes) that embeddings blur. RRF fuses the two rank lists without needing to
  normalize wildly different score scales; the cross-encoder reranker then reorders the
  fused top candidates using the actual query-passage pair, which neither dense similarity
  nor BM25 can do alone.
- **Company scoping**: every store call takes a `company_id` — Pinecone namespace, a
  per-company BM25 index on disk, a per-company local-store directory. Multi-tenant by
  construction, not retrofitted.
- **Free-tier friendly**: embeddings are cached by content hash (`.indexes/embedding_cache`)
  so re-ingesting unchanged docs costs nothing; the reranker is a local ~34MB ONNX model
  (no API call, no torch); generation calls are the only per-query API cost.
- **Two response modes off one retrieval path**: `answer` and `prompt` share the exact same
  retrieval + prompt-assembly code, so they can never disagree about what was retrieved —
  only about whether Gemini is then called to generate from it.
- **Auth is authorization, not just authentication**: an API key doesn't just prove you're a
  known caller, it carries the specific `company_id`s you're allowed to touch. Without this,
  `company_id` would be a client-supplied field with no enforcement — any caller could read or
  pollute another tenant's document index just by naming it in the request.
