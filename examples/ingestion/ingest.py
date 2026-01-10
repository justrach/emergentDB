#!/usr/bin/env python3
"""
EmergentDB Document Ingestion CLI

A CLI tool for ingesting documents into EmergentDB using Gemini for OCR and embeddings.
The embeddings are stored in the Rust EmergentDB server for optimized vector search.

Usage:
    python ingest.py ingest document.pdf
    python ingest.py ingest-dir ./documents
    python ingest.py query "What is machine learning?"
    python ingest.py list
    python ingest.py interactive
"""

import os
import sys
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from dotenv import load_dotenv
import requests

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

console = Console()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMERGENT_API_URL = os.getenv("EMERGENT_API_URL", "http://localhost:3000")
DB_PATH = Path(__file__).parent / "metadata.db"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks


class MetadataDB:
    """Local SQLite for metadata only (text content, filenames). Vectors go to EmergentDB."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT,
                content TEXT,
                chunk_index INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(id)
        """)
        self.conn.commit()

    def insert(self, doc_id: int, filename: str, filepath: str, content: str,
               chunk_index: int, metadata: Dict = None):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (id, filename, filepath, content, chunk_index, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, filename, filepath, content, chunk_index, json.dumps(metadata or {})))
        self.conn.commit()

    def get_by_id(self, doc_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename, content, metadata FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "filename": row[1],
                "content": row[2],
                "metadata": json.loads(row[3])
            }
        return None

    def get_by_ids(self, doc_ids: List[int]) -> Dict[int, Dict]:
        """Get multiple documents by ID."""
        if not doc_ids:
            return {}
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(doc_ids))
        cursor.execute(f"SELECT id, filename, content, metadata FROM documents WHERE id IN ({placeholders})", doc_ids)
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                "id": row[0],
                "filename": row[1],
                "content": row[2],
                "metadata": json.loads(row[3])
            }
        return results

    def list_documents(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT filename, COUNT(*) as chunks, MIN(created_at) as created
            FROM documents GROUP BY filename ORDER BY created DESC
        """)
        return [{"filename": r[0], "chunks": r[1], "created": r[2]} for r in cursor.fetchall()]

    def delete_document(self, filename: str) -> List[int]:
        """Delete document and return the IDs that were deleted."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        ids = [row[0] for row in cursor.fetchall()]
        cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        self.conn.commit()
        return ids

    def clear(self) -> List[int]:
        """Clear all and return IDs that were deleted."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents")
        ids = [row[0] for row in cursor.fetchall()]
        cursor.execute("DELETE FROM documents")
        self.conn.commit()
        return ids

    def stats(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_chunks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
        total_docs = cursor.fetchone()[0]
        return {"total_documents": total_docs, "total_chunks": total_chunks}

    def close(self):
        self.conn.close()


class EmergentDBClient:
    """Client for the Rust EmergentDB API server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict:
        """Check server health."""
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def insert_vector(self, doc_id: int, embedding: List[float], metadata: Dict = None) -> Dict:
        """Insert a vector into EmergentDB."""
        resp = requests.post(
            f"{self.base_url}/vectors/insert",
            json={"id": doc_id, "vector": embedding, "metadata": metadata or {}},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def batch_insert(self, vectors: List[Dict]) -> Dict:
        """Batch insert vectors."""
        resp = requests.post(
            f"{self.base_url}/vectors/batch_insert",
            json={"vectors": vectors},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        resp = requests.post(
            f"{self.base_url}/vectors/search",
            json={"query": query_embedding, "k": k},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def stats(self) -> Dict:
        """Get ingestion stats from server."""
        resp = requests.get(f"{self.base_url}/ingest/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()


class GeminiClient:
    """Client for Gemini API - OCR and embeddings."""

    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        self.client = genai.Client(api_key=api_key)

    def extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from image using Gemini vision."""
        with open(image_path, "rb") as f:
            image_data = f.read()

        mime_type = self._get_mime_type(image_path)

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text="Extract all text from this image. Return only the extracted text, no commentary."),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=image_data,
                            )
                        )
                    ]
                )
            ]
        )

        return response.text

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyPDF2 or Gemini OCR."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            # If PDF has little text, it might be scanned - use OCR
            if len(full_text.strip()) < 100:
                console.print("[yellow]PDF appears to be scanned, using Gemini OCR...[/yellow]")
                return self._ocr_pdf(pdf_path)

            return full_text

        except Exception as e:
            console.print(f"[yellow]PDF text extraction failed, using OCR: {e}[/yellow]")
            return self._ocr_pdf(pdf_path)

    def _ocr_pdf(self, pdf_path: Path) -> str:
        """OCR a PDF using Gemini vision."""
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text="Extract all text from this PDF document. Return only the extracted text, preserving the structure and formatting as much as possible."),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=pdf_data,
                            )
                        )
                    ]
                )
            ]
        )

        return response.text

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini embedding model."""
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        return list(result.embeddings[0].values)

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a search query."""
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        return list(result.embeddings[0].values)

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for a file."""
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_types.get(suffix, "application/octet-stream")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def generate_doc_id(filename: str, chunk_index: int) -> int:
    """Generate unique document ID as integer."""
    content = f"{filename}:{chunk_index}"
    # Use first 8 bytes of MD5 hash as int
    hash_bytes = hashlib.md5(content.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big') % (2**63)  # Keep it positive


def check_server(client: EmergentDBClient) -> bool:
    """Check if EmergentDB server is running."""
    try:
        client.health()
        return True
    except Exception:
        return False


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """EmergentDB Document Ingestion CLI

    Ingest documents into EmergentDB using Gemini for OCR and embeddings.
    Vectors are stored in the Rust EmergentDB server for optimized search.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--chunk-size", default=CHUNK_SIZE, help="Characters per chunk")
@click.option("--overlap", default=CHUNK_OVERLAP, help="Overlap between chunks")
def ingest(filepath: str, chunk_size: int, overlap: int):
    """Ingest a single document into EmergentDB."""
    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY not set in .env file[/red]")
        sys.exit(1)

    path = Path(filepath)
    emergent = EmergentDBClient(EMERGENT_API_URL)

    # Check server is running
    if not check_server(emergent):
        console.print(f"[red]Error: EmergentDB server not running at {EMERGENT_API_URL}[/red]")
        console.print("[yellow]Start it with: cargo run --release -p api-server[/yellow]")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {path.name}...", total=None)

        gemini = GeminiClient(GEMINI_API_KEY)
        metadata_db = MetadataDB(DB_PATH)

        try:
            # Extract text based on file type
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                progress.update(task, description="Extracting text from PDF...")
                text = gemini.extract_text_from_pdf(path)
            elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                progress.update(task, description="Running OCR on image...")
                text = gemini.extract_text_from_image(path)
            elif suffix in [".txt", ".md", ".json", ".csv"]:
                progress.update(task, description="Reading text file...")
                text = path.read_text(encoding="utf-8")
            else:
                console.print(f"[yellow]Unsupported file type: {suffix}[/yellow]")
                return

            if not text.strip():
                console.print("[yellow]No text extracted from document[/yellow]")
                return

            # Chunk the text
            progress.update(task, description="Chunking text...")
            chunks = chunk_text(text, chunk_size, overlap)
            console.print(f"[dim]Created {len(chunks)} chunks[/dim]")

            # Generate embeddings and store in EmergentDB
            progress.update(task, description="Generating embeddings...")
            for i, chunk in enumerate(chunks):
                doc_id = generate_doc_id(path.name, i)

                progress.update(task, description=f"Embedding chunk {i+1}/{len(chunks)}...")
                embedding = gemini.get_embedding(chunk)

                # Store vector in EmergentDB (Rust server)
                progress.update(task, description=f"Storing chunk {i+1}/{len(chunks)} in EmergentDB...")
                emergent.insert_vector(doc_id, embedding, {"filename": path.name, "chunk": i})

                # Store metadata locally
                metadata_db.insert(
                    doc_id=doc_id,
                    filename=path.name,
                    filepath=str(path.absolute()),
                    content=chunk,
                    chunk_index=i,
                    metadata={"chunk_index": i, "total_chunks": len(chunks)}
                )

            metadata_db.close()

        except requests.exceptions.ConnectionError:
            console.print(f"[red]Error: Cannot connect to EmergentDB server at {EMERGENT_API_URL}[/red]")
            raise
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise

    console.print(f"[green]Successfully ingested {path.name} ({len(chunks)} chunks) into EmergentDB[/green]")


@cli.command()
@click.argument("dirpath", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Process subdirectories")
def ingest_dir(dirpath: str, recursive: bool):
    """Ingest all documents in a directory."""
    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY not set in .env file[/red]")
        sys.exit(1)

    path = Path(dirpath)
    extensions = [".pdf", ".txt", ".md", ".jpg", ".jpeg", ".png", ".gif", ".webp"]
    pattern = "**/*" if recursive else "*"

    files = []
    for ext in extensions:
        files.extend(path.glob(f"{pattern}{ext}"))

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    console.print(f"Found {len(files)} files to process")

    for file_path in files:
        console.print(f"\n[bold]Processing: {file_path.name}[/bold]")
        ctx = click.Context(ingest)
        ctx.invoke(ingest, filepath=str(file_path))


@cli.command()
@click.argument("query_text")
@click.option("--k", default=5, help="Number of results to return")
def query(query_text: str, k: int):
    """Search EmergentDB for similar documents."""
    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY not set in .env file[/red]")
        sys.exit(1)

    emergent = EmergentDBClient(EMERGENT_API_URL)

    if not check_server(emergent):
        console.print(f"[red]Error: EmergentDB server not running at {EMERGENT_API_URL}[/red]")
        console.print("[yellow]Start it with: cargo run --release -p api-server[/yellow]")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)

        gemini = GeminiClient(GEMINI_API_KEY)
        metadata_db = MetadataDB(DB_PATH)

        # Generate query embedding
        progress.update(task, description="Generating query embedding...")
        query_embedding = gemini.get_query_embedding(query_text)

        # Search EmergentDB (Rust server)
        progress.update(task, description="Searching EmergentDB...")
        results = emergent.search(query_embedding, k=k)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            metadata_db.close()
            return

        # Get metadata for results
        doc_ids = [int(r["id"]) for r in results]
        metadata_map = metadata_db.get_by_ids(doc_ids)
        metadata_db.close()

    console.print(f"\n[bold]Found {len(results)} results from EmergentDB:[/bold]\n")

    for i, result in enumerate(results, 1):
        doc_id = int(result["id"])
        score = result["score"]
        meta = metadata_map.get(doc_id, {})

        filename = meta.get("filename", f"ID: {doc_id}")
        content = meta.get("content", "[Content not found in metadata]")

        console.print(Panel(
            f"[dim]Score: {score:.4f}[/dim]\n\n{content[:500]}{'...' if len(content) > 500 else ''}",
            title=f"[bold cyan]{i}. {filename}[/bold cyan]",
            border_style="blue"
        ))


@cli.command("list")
def list_docs():
    """List all ingested documents."""
    if not DB_PATH.exists():
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return

    metadata_db = MetadataDB(DB_PATH)
    documents = metadata_db.list_documents()
    metadata_db.close()

    if not documents:
        console.print("[yellow]No documents found.[/yellow]")
        return

    # Also show server stats
    emergent = EmergentDBClient(EMERGENT_API_URL)
    server_stats = None
    if check_server(emergent):
        try:
            server_stats = emergent.stats()
        except Exception:
            pass

    table = Table(title="Ingested Documents")
    table.add_column("Filename", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Created", style="dim")

    for doc in documents:
        table.add_row(doc["filename"], str(doc["chunks"]), doc["created"])

    console.print(table)

    if server_stats:
        console.print(f"\n[dim]EmergentDB server: {server_stats['total_documents']} vectors stored[/dim]")


@cli.command()
@click.argument("filename")
def delete(filename: str):
    """Delete a document by filename."""
    if not DB_PATH.exists():
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return

    metadata_db = MetadataDB(DB_PATH)
    deleted_ids = metadata_db.delete_document(filename)
    metadata_db.close()

    if deleted_ids:
        console.print(f"[green]Deleted {len(deleted_ids)} chunks from '{filename}'[/green]")
        console.print("[yellow]Note: Vectors remain in EmergentDB server (restart server to clear)[/yellow]")
    else:
        console.print(f"[yellow]No document found with filename '{filename}'[/yellow]")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to clear all documents?")
def clear():
    """Clear all documents from the database."""
    if not DB_PATH.exists():
        console.print("[yellow]Database is already empty.[/yellow]")
        return

    metadata_db = MetadataDB(DB_PATH)
    deleted_ids = metadata_db.clear()
    metadata_db.close()

    console.print(f"[green]Cleared {len(deleted_ids)} chunks from metadata.[/green]")
    console.print("[yellow]Note: Restart EmergentDB server to clear vectors[/yellow]")


@cli.command()
def stats():
    """Show database statistics."""
    metadata_db = MetadataDB(DB_PATH) if DB_PATH.exists() else None
    local_stats = metadata_db.stats() if metadata_db else {"total_documents": 0, "total_chunks": 0}
    if metadata_db:
        metadata_db.close()

    emergent = EmergentDBClient(EMERGENT_API_URL)
    server_online = check_server(emergent)
    server_stats = emergent.stats() if server_online else None

    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Local Documents", str(local_stats["total_documents"]))
    table.add_row("Local Chunks", str(local_stats["total_chunks"]))
    table.add_row("", "")

    if server_online and server_stats:
        table.add_row("EmergentDB Status", "[green]Online[/green]")
        table.add_row("EmergentDB Vectors", str(server_stats["total_documents"]))
        table.add_row("Vector Dimension", str(server_stats["vector_dimension"]))
        table.add_row("Index Type", server_stats["index_type"])
    else:
        table.add_row("EmergentDB Status", "[red]Offline[/red]")

    table.add_row("", "")
    table.add_row("Server URL", EMERGENT_API_URL)

    console.print(table)


@cli.command()
def interactive():
    """Start interactive mode."""
    if not GEMINI_API_KEY:
        console.print("[red]Error: GEMINI_API_KEY not set in .env file[/red]")
        sys.exit(1)

    emergent = EmergentDBClient(EMERGENT_API_URL)
    server_status = "[green]Online[/green]" if check_server(emergent) else "[red]Offline[/red]"

    console.print(Panel(
        f"[bold]EmergentDB Interactive Mode[/bold]\n\n"
        f"Server: {EMERGENT_API_URL} ({server_status})\n\n"
        "Commands:\n"
        "  [cyan]ingest <filepath>[/cyan] - Ingest a document\n"
        "  [cyan]query <text>[/cyan] - Search EmergentDB\n"
        "  [cyan]list[/cyan] - List all documents\n"
        "  [cyan]stats[/cyan] - Show statistics\n"
        "  [cyan]exit[/cyan] - Exit interactive mode",
        title="Welcome",
        border_style="green"
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]>[/bold green] ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "exit" or cmd == "quit":
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "ingest":
                if not arg:
                    console.print("[yellow]Usage: ingest <filepath>[/yellow]")
                else:
                    ctx = click.Context(ingest)
                    ctx.invoke(ingest, filepath=arg)
            elif cmd == "query":
                if not arg:
                    console.print("[yellow]Usage: query <search text>[/yellow]")
                else:
                    ctx = click.Context(query)
                    ctx.invoke(query, query_text=arg)
            elif cmd == "list":
                ctx = click.Context(list_docs)
                ctx.invoke(list_docs)
            elif cmd == "stats":
                ctx = click.Context(stats)
                ctx.invoke(stats)
            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()
