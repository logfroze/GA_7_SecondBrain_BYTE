import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from src.rag_pipeline import SecondBrainRAG
from src.config import validate_config

console = Console()


def display_banner():
    console.print(
        Panel(
            "[bold cyan]GA_7_SecondBrain_BYTE[/bold cyan]\n"
            "[white]Local RAG System with Page-Accurate Citations & Local Embeddings[/white]\n"
            "[dim]BYTE Arithmatrix Generative AI Internship — Task 7[/dim]",
            border_style="cyan",
            expand=False,
        )
    )


def handle_index(rag: SecondBrainRAG, pdf_path: str):
    console.print(f"\n[yellow]⏳ Reading and indexing:[/yellow] [bold]{pdf_path}[/bold]...")
    try:
        summary = rag.index_pdf(pdf_path)
        console.print(f"[green]✓ Successfully indexed '{summary['source']}'[/green]")
        console.print(f"  • Pages: [bold]{summary['total_pages']}[/bold]")
        console.print(f"  • Chunks generated: [bold]{summary['total_chunks']}[/bold]")
        console.print(f"  • Vectors stored: [bold]{summary['indexed_count']}[/bold]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Indexing error:[/bold red] {e}")


def handle_query(rag: SecondBrainRAG, query: str):
    console.print(f"\n[yellow]🔍 Question:[/yellow] [bold]{query}[/bold]\n")
    with console.status("[cyan]Retrieving context and generating grounded answer...[/cyan]"):
        try:
            result = rag.ask(query)
        except Exception as e:
            console.print(f"[bold red]❌ Query error:[/bold red] {e}")
            return

    # Print Answer
    console.print(
        Panel(
            result["answer"],
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )

    # Print Citations & Sources Table
    if result["citations"]:
        citations_str = ", ".join(f"Page {p}" for p in result["citations"])
        console.print(f"[bold cyan]📄 Verified Citations:[/bold cyan] {citations_str}\n")

    if result["sources"]:
        table = Table(title="Retrieved Context Excerpts", show_lines=True)
        table.add_column("Page", style="cyan", justify="center", width=8)
        table.add_column("Source", style="dim", width=20)
        table.add_column("Excerpt Preview", style="white")

        for s in result["sources"]:
            table.add_row(
                str(s.get("page", "?")),
                str(s.get("source", "")),
                s.get("text_preview", ""),
            )
        console.print(table)
        console.print("")


def interactive_mode(rag: SecondBrainRAG):
    display_banner()
    console.print("[dim]Type 'index <path_to_pdf>' to load a file, 'exit' or 'quit' to close.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]SecondBrain[/bold blue]")
            if not user_input or not user_input.strip():
                continue

            cleaned = user_input.strip()
            if cleaned.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if cleaned.lower().startswith("index "):
                path_str = cleaned[6:].strip().strip("\"'")
                handle_index(rag, path_str)
            elif cleaned.lower() == "clear":
                rag.reset_knowledge_base()
                console.print("[green]✓ Vector store collection cleared.[/green]")
            else:
                handle_query(rag, cleaned)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/dim]")
            break


def main():
    parser = argparse.ArgumentParser(description="Second Brain Local RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a PDF document")
    index_parser.add_argument("--pdf", required=True, help="Path to the PDF document")

    # Query command
    query_parser = subparsers.add_parser("ask", help="Query the indexed document")
    query_parser.add_argument("--query", required=True, help="Question to ask")

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive terminal session")

    args = parser.parse_args()

    # Validate configuration
    try:
        validate_config()
    except ValueError as ve:
        console.print(f"[bold red]Configuration Error:[/bold red] {ve}")
        sys.exit(1)

    rag = SecondBrainRAG()

    if args.command == "index":
        handle_index(rag, args.pdf)
    elif args.command == "ask":
        handle_query(rag, args.query)
    elif args.command == "interactive" or args.command is None:
        interactive_mode(rag)


if __name__ == "__main__":
    main()
