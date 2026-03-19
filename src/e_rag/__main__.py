"""RAG pipeline demo entrypoint."""

from pathlib import Path

import typer
from rich.console import Console

from e_rag.core.settings import settings as st
from e_rag.core.stores import create_child_store, create_parent_store
from e_rag.pipelines.indexing import run_indexing
from e_rag.pipelines.query import build_query_pipeline

console = Console()


def run(
    source: str = typer.Option(
        str(st.BASE_DIR / "pdf_sample_full.pdf"),
        help="PDF file path to index",
    ),
    query: str = typer.Option(
        "What is this document about?",
        help="Query to run against indexed documents",
    ),
    recreate: bool = typer.Option(False, help="Recreate database tables"),
) -> None:
    """Index a PDF and run hybrid retrieval with parent document resolution."""
    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]File not found: {source}[/red]")
        raise typer.Exit(1)

    parent_store = create_parent_store(recreate=recreate)
    child_store = create_child_store(recreate=recreate)

    console.print(f"[bold]Indexing:[/bold] {source_path.name}")
    stats = run_indexing(
        sources=[str(source_path)],
        parent_store=parent_store,
        child_store=child_store,
    )
    console.print(
        f"  Parents: {stats['parents']}  Children: {stats['children']}"
    )

    console.print(f"\n[bold]Query:[/bold] {query}")
    pipeline = build_query_pipeline(
        child_store=child_store,
        parent_store=parent_store,
    )
    result = pipeline.run(
        {
            "text_embedder": {"text": query},
            "keyword_retriever": {"query": query},
        }
    )

    documents = result["parent_retriever"]["documents"]
    console.print(f"\n[bold]Retrieved {len(documents)} parent documents:[/bold]\n")
    for i, doc in enumerate(documents, 1):
        console.print(f"[cyan]--- Document {i} ---[/cyan]")
        content = doc.content[:500] if doc.content else "[empty]"
        console.print(content)
        console.print()


app = typer.Typer(name="e-rag")
app.command()(run)

if __name__ == "__main__":
    app()
