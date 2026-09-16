#!/bin/sh
set -e

COMPANY="${PUBLIC_CHAT_COMPANY_ID:-corvit}"

echo "Running initial sync ingest for company '$COMPANY'..."
python -m rag.cli ingest "data/$COMPANY" --company "$COMPANY" --sync || \
    echo "Initial ingest failed (check GEMINI_API_KEY / PINECONE_API_KEY secrets) -- starting server anyway."

exec uvicorn rag.api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
