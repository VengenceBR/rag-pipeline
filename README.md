---
title: Corvit Assistant
emoji: 💬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

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

## Keeping the index in sync automatically

`ingest` defaults to `--sync`: it treats the directory as the company's *complete* current doc
set, so a file that's been deleted (or edited into fewer chunks) has its old chunks removed from
the index, not just its new content added. Pass `--no-sync` for incremental/partial uploads
(this is what `POST /ingest` uses, since it only ever receives whatever files were attached to
that one request).

To auto re-ingest whenever files change, run a watcher instead of calling `ingest` by hand:

```bash
python -m rag.cli watch data/sample_company --company acme          # one company
python -m rag.cli watch-all --data-dir data                          # every data/<company_id>/ subdir
```

Either runs forever, does an initial sync on startup, then re-ingests (always in sync mode) after
`--debounce` seconds (default 3) of quiet following the last change under the watched directory —
so a bulk copy or an editor's save-then-rename collapses into one re-ingest, not several. `docker
compose` ships this as an optional `rag-watcher` service alongside `rag-api`, sharing the same
`data/` and index volumes.

## Run the API

```bash
uvicorn rag.api.main:app --reload
```

Every route except `/health` requires an `X-API-Key` header, scoped to specific `company_id`s
so one client can never query or ingest into another company's index. Manage keys with the CLI:

```bash
python -m rag.cli keys create --name "acme support bot" --company acme
python -m rag.cli keys create --name "acme support bot" --company acme --rate-limit 60
python -m rag.cli keys list
python -m rag.cli keys revoke sk-...
```

Set `REQUIRE_API_KEY=false` in `.env` to disable auth (and rate limiting) entirely (local dev only).

Each key is also rate-limited (`RATE_LIMIT_PER_MINUTE` in `.env`, default 20/min; override per key with
`--rate-limit`) — a sliding window keyed by the API key, so one leaked or misbehaving key can't burn
through the whole Gemini free-tier quota. Over the limit returns `429` with a `Retry-After` header.
This is in-process, so it resets on restart and doesn't share state across multiple replicas — fine
for the single-container deployment this ships with; swap in a Redis-backed limiter if you scale out.

- `GET /health` — no key required
- `POST /ingest?company_id=acme` (multipart file upload)
- `POST /query` `{"query": "...", "company_id": "acme"}` → grounded answer + citations
- `POST /prompt` `{"query": "...", "company_id": "acme"}` → assembled prompt package, no generation

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: sk-..." -H "Content-Type: application/json" \
  -d '{"query": "how many days do I have to return an item?", "company_id": "acme"}'
```

## Two frontends, two trust models

- **`GET /`** — the public chat site (`src/rag/api/static/index.html`). No API key, ever. It only
  ever talks to `POST /chat`, which is hardcoded to one company (`PUBLIC_CHAT_COMPANY_ID`, default
  `corvit`) and rate-limited per client IP instead of per key. This is the pattern for anything
  actual visitors will load in a browser — the request never carries a credential that could leak.
- **`GET /demo`** (`src/rag/api/static/demo.html`) — a debug tool for *you*: it takes a real
  `X-API-Key` and lets you hit any company/mode by hand. Fine for local testing, **never** meant
  to be the thing you point real users at, since the key sits in the page.

## Embedding the widget on an existing site

To drop the chat bubble onto a site this project doesn't otherwise control (e.g. Corvit's
actual marketing site), the site owner adds exactly one line, anywhere in the page:

```html
<script src="https://<your-deployed-domain>/widget-loader.js" async></script>
```

That's the whole integration. The loader script derives the backend's origin from its own
`<script src>`, injects a floating bubble button, and opens a same-origin iframe pointing back
at `/` when clicked — so it never touches the host page's CSS/JS, and never needs CORS (the
iframe's own requests to `/chat` are same-origin to *us*, not to the host page). Swap
`src/rag/api/static/widget-loader.js`'s bubble color/size, or `index.html`'s branding, to match
whatever site it's going on.

### WordPress: `wordpress-plugin/corvit-ai-assistant/`

For a WordPress-based site (corvit.com runs WordPress), raw script injection usually isn't the
preferred integration path — site owners generally want add-ons managed through the plugin
system rather than a theme-file edit. `wordpress-plugin/corvit-ai-assistant/` is a small plugin
that does exactly the same thing as the snippet above, the WordPress-native way:

- `Settings → Corvit AI Assistant` in wp-admin — set the backend URL once, toggle the widget
  on/off. The widget script is not enqueued at all until a URL is configured.
- Enqueues `widget-loader.js` via `wp_enqueue_script()` on `wp_enqueue_scripts` (frontend only,
  never in `wp-admin`), loaded `async` so it can't block page render.
- No theme files touched, no data stored in WordPress — the plugin's only job is getting that
  one script tag onto the page.

Install via `Plugins → Add New → Upload Plugin` with `wordpress-plugin/corvit-ai-assistant.zip`
(rebuild it after editing the plugin source: zip the `corvit-ai-assistant/` folder so the zip's
top-level entry is the folder itself, matching what WordPress's uploader expects).

## Deploying (free): Hugging Face Spaces

The root `Dockerfile` (not `docker/Dockerfile`, which is for local `docker compose`) targets
[Spaces' Docker SDK](https://huggingface.co/docs/hub/spaces-sdks-docker): runs as the required
uid 1000, listens on port 7860, and `entrypoint.sh` re-ingests `data/<PUBLIC_CHAT_COMPANY_ID>`
in sync mode on every boot before starting the server — Spaces' disk doesn't persist across
restarts on the free tier, so the index needs to be rebuildable from what's committed in `data/`.

1. Create a Space at huggingface.co/new-space with SDK = Docker (the `README.md` YAML frontmatter
   here already sets `sdk: docker` / `app_port: 7860`, so a fresh Space using this repo's content
   picks them up automatically).
2. In the Space's Settings → Repository secrets, add `GEMINI_API_KEY` and `PINECONE_API_KEY`
   (`PINECONE_INDEX_NAME` too, if not using the default). Secrets, not public Variables — they're
   injected as env vars at runtime and never appear in the Space's files.
3. Push this repo to the Space's git remote. It builds and serves `/` at the Space's URL.

Free CPU Spaces need no credit card. Cold boot re-embeds `data/corvit`'s few chunks (trivial,
free-tier embedding quota is generous) and re-populates BM25; Pinecone itself is external and
already persists regardless.

## Tests

```bash
pytest
```

The full suite runs with no API keys set — it exercises chunking, RRF fusion, the local
vector store, BM25 (including sync-deletion and the empty-corpus edge case), API auth
(401/403/429/200 paths), and the watcher's debounce logic directly, plus an end-to-end
retrieval pass with a fake embedder.

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
- **Re-ingestion is a sync, not just an append**: `watch`/`watch-all` (and `ingest --sync`, the
  default) diff against a persisted per-company manifest, so a doc that's deleted or shrinks
  during an edit actually loses its stale chunks from the vector store *and* the BM25 index —
  not just gains new ones. A watcher that only ever adds would let deleted docs answer queries
  forever.
