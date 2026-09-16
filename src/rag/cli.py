from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import typer

from rag.config import settings

app = typer.Typer(add_completion=False, help="Hybrid RAG pipeline CLI")
keys_app = typer.Typer(add_completion=False, help="Manage API keys for the FastAPI server")
app.add_typer(keys_app, name="keys")


@contextmanager
def _friendly_errors():
    try:
        yield
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest(
    directory: Path = typer.Argument(..., help="Directory of company documents to ingest"),
    company: str = typer.Option(..., "--company", "-c", help="Company/tenant id"),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Treat DIRECTORY as this company's complete doc set: chunks for files no "
        "longer there (deleted, or edited into fewer chunks) are removed from the index. "
        "Use --no-sync to only add/update, e.g. when DIRECTORY is just a partial batch.",
    ),
):
    """Load, chunk, embed, and index a directory of documents for one company."""
    from rag.service import RAGService

    with _friendly_errors():
        result = RAGService().ingest(directory, company, sync=sync)
        typer.echo(result.model_dump_json(indent=2))


@app.command()
def query(
    text: str = typer.Argument(..., help="The question to ask"),
    company: str = typer.Option(..., "--company", "-c", help="Company/tenant id"),
):
    """Retrieve + generate a grounded, cited answer."""
    from rag.service import RAGService

    with _friendly_errors():
        response = RAGService().query(text, company, mode="answer")
        typer.echo(f"\nAnswer:\n{response.answer}\n")
        typer.echo("Citations:")
        for c in response.citations:
            loc = f", p.{c.page}" if c.page else ""
            typer.echo(f"  {c.marker} {c.title}{loc} ({c.source_path})")


@app.command()
def prompt(
    text: str = typer.Argument(..., help="The question to build a prompt for"),
    company: str = typer.Option(..., "--company", "-c", help="Company/tenant id"),
):
    """Retrieve context and print the assembled prompt package WITHOUT generating."""
    from rag.service import RAGService

    with _friendly_errors():
        response = RAGService().query(text, company, mode="prompt")
        typer.echo(response.prompt_package.model_dump_json(indent=2))


@app.command()
def search(
    text: str = typer.Argument(..., help="Query to test retrieval quality for"),
    company: str = typer.Option(..., "--company", "-c", help="Company/tenant id"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
):
    """Run hybrid retrieval only, and print ranked chunks with scores. For debugging."""
    from rag.service import RAGService

    with _friendly_errors():
        results = RAGService().retriever.retrieve(text, company, top_k=top_k)
        for i, rc in enumerate(results, start=1):
            location = (
                f"{rc.chunk.title} (p.{rc.chunk.page})" if rc.chunk.page else rc.chunk.title
            )
            typer.echo(
                f"[{i}] rerank={rc.rerank_score:.4f} fused={rc.fused_score:.4f} "
                f"dense={rc.dense_score} sparse={rc.sparse_score}  {location}"
            )
            typer.echo(f"    {rc.chunk.text[:200]}...")


@app.command()
def watch(
    directory: Path = typer.Argument(..., help="Directory to watch for one company"),
    company: str = typer.Option(..., "--company", "-c", help="Company/tenant id"),
    debounce: float = typer.Option(
        3.0, "--debounce", help="Seconds of quiet after a change before re-ingesting"
    ),
):
    """Watch a directory and auto re-ingest (sync mode) on every change. Runs forever."""
    from rag.watch import watch_companies

    with _friendly_errors():
        typer.echo(f"Watching {directory} for company '{company}' (Ctrl+C to stop)...")
        watch_companies({company: directory}, debounce_seconds=debounce)


@app.command("watch-all")
def watch_all(
    data_dir: Path = typer.Option(
        settings.data_dir,
        "--data-dir",
        help="Each immediate subdirectory is watched as its own company (folder name = company_id)",
    ),
    debounce: float = typer.Option(
        3.0, "--debounce", help="Seconds of quiet after a change before re-ingesting"
    ),
):
    """Watch every company subdirectory under DATA_DIR and auto re-ingest on change. Runs forever."""
    from rag.watch import discover_company_dirs, watch_companies

    with _friendly_errors():
        company_dirs = discover_company_dirs(data_dir)
        if not company_dirs:
            typer.echo(f"No company subdirectories found under {data_dir}.", err=True)
            raise typer.Exit(code=1)
        typer.echo(
            f"Watching {len(company_dirs)} companies under {data_dir}: "
            f"{list(company_dirs)} (Ctrl+C to stop)..."
        )
        watch_companies(company_dirs, debounce_seconds=debounce)


@app.command()
def stats(company: str = typer.Option(..., "--company", "-c", help="Company/tenant id")):
    """Show how many chunks are currently indexed for a company."""
    manifest_path = settings.index_dir / "manifest" / f"{company}.json"
    if not manifest_path.exists():
        typer.echo(f"No data indexed for company '{company}' yet.")
        raise typer.Exit(code=1)
    chunks = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc_ids = {c["doc_id"] for c in chunks}
    typer.echo(f"company={company}  documents={len(doc_ids)}  chunks={len(chunks)}")
    typer.echo(f"vector store: {'pinecone' if settings.use_pinecone else 'local (numpy)'}")


@app.command("check-models")
def check_models():
    """Verify the configured Gemini model IDs are reachable with the current API key."""
    if not settings.use_gemini:
        typer.echo("GEMINI_API_KEY is not set.")
        raise typer.Exit(code=1)

    from google import genai

    client = genai.Client(api_key=settings.gemini_api_key)
    available = {m.name.split("/")[-1] for m in client.models.list()}

    for label, model_id in (
        ("embedding", settings.embedding_model),
        ("generation", settings.generation_model),
    ):
        status = "OK" if model_id in available else "NOT FOUND for this key"
        typer.echo(f"{label}: {model_id} -> {status}")


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", "-n", help="Human-readable label for this key"),
    company: list[str] = typer.Option(
        ...,
        "--company",
        "-c",
        help="Company/tenant id this key may access. Repeat for multiple, or pass '*' for all.",
    ),
    rate_limit: int | None = typer.Option(
        None,
        "--rate-limit",
        "-r",
        help="Requests/minute for this key. Defaults to RATE_LIMIT_PER_MINUTE if unset.",
    ),
):
    """Generate a new API key for the FastAPI server, scoped to one or more companies."""
    from rag.auth import create_api_key

    new_key = create_api_key(name=name, companies=company, rate_limit_per_minute=rate_limit)
    typer.echo(f"Created key for '{name}' (companies={company}, rate_limit={rate_limit}):\n\n  {new_key}\n")
    typer.echo("Store this now - it is not re-displayed. Send it as the X-API-Key header.")


@keys_app.command("list")
def keys_list():
    """List existing API keys (names, authorized companies, and rate limits — not the key values)."""
    from rag.auth import load_api_keys
    from rag.config import settings

    records = load_api_keys()
    if not records:
        typer.echo("No API keys yet. Create one with: rag keys create --name ... --company ...")
        return
    for key, record in records.items():
        masked = f"{key[:7]}...{key[-4:]}"
        limit = record.rate_limit_per_minute or f"{settings.rate_limit_per_minute} (default)"
        typer.echo(
            f"{masked}  name={record.name}  companies={record.companies}  rate_limit={limit}/min"
        )


@keys_app.command("revoke")
def keys_revoke(api_key: str = typer.Argument(..., help="Full API key to revoke")):
    """Revoke an API key."""
    from rag.auth import revoke_api_key

    if revoke_api_key(api_key):
        typer.echo("Key revoked.")
    else:
        typer.echo("Key not found.", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
