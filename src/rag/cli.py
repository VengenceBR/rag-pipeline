from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import typer

from rag.config import settings

app = typer.Typer(add_completion=False, help="Hybrid RAG pipeline CLI")


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
):
    """Load, chunk, embed, and index a directory of documents for one company."""
    from rag.service import RAGService

    with _friendly_errors():
        result = RAGService().ingest(directory, company)
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


if __name__ == "__main__":
    app()
