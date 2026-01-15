from __future__ import annotations

from pathlib import Path
import typer
import json

from .config import VaultConfig
from .indexer.indexer import Indexer
from .retrieval.retriever import Retriever
from .mcp.server import run_stdio_server

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _cfg(config: str) -> VaultConfig:
    return VaultConfig.from_toml(config)

@app.command()
def init(vault: str = typer.Option(..., help="Vault root path"),
         index: str = typer.Option(..., help="Index directory"),
         out: str = typer.Option("config.toml", help="Write example config to this path")):
    """Write a starter config.toml."""
    outp = Path(out)
    outp.write_text(f"""[vault]
root = "{vault}"
ignore = [".git/**", ".obsidian/cache/**", "**/.DS_Store"]

[index]
dir = "{index}"
extractor_version = "v1"
chunker_version = "v1"

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
batch_size = 32
device = "cpu"

[ocr]
mode = "off"

[retrieval]
k_vec = 40
k_lex = 40
top_k = 10
use_rerank = false

[mcp]
transport = "stdio"
""", encoding="utf-8")
    typer.echo(f"Wrote {outp}")

@app.command()
def scan(config: str = typer.Option("config.toml"), full: bool = typer.Option(False, help="Re-index all files")):
    """Scan vault and index."""
    cfg = _cfg(config)
    idx = Indexer(cfg)
    idx.scan(full=full)
    typer.echo("Scan complete.")

@app.command()
def query(q: str, config: str = typer.Option("config.toml"), k: int = typer.Option(10),
          path: str = typer.Option("", help="Path prefix filter")):
    """Search the vault."""
    cfg = _cfg(config)
    r = Retriever(cfg)
    filters = {"vault_id": r.store.status(r.cfg.index_dir.name if False else "")}  # placeholder
    filt = {}
    if path:
        filt["path_prefix"] = path
    # In skeleton, vault_id is computed by Indexer; for now omit unless you set it explicitly.
    hits = r.search(q, k=k, filters=filt)
    typer.echo(json.dumps([{
        "chunk_id": h.chunk_id,
        "score": h.score,
        "snippet": h.snippet,
        "source_ref": h.source_ref.__dict__,
        "metadata": h.metadata,
    } for h in hits], indent=2))

@app.command()
def open(chunk_id: str, config: str = typer.Option("config.toml")):
    """Open a chunk by ID (skeleton)."""
    cfg = _cfg(config)
    r = Retriever(cfg)
    from .models import SourceRef
    sr = SourceRef(vault_id="", rel_path="", file_type="unknown", anchor_type="chunk", anchor_ref=chunk_id, locator={})
    out = r.open(sr)
    typer.echo(out.content)

@app.command()
def watch(config: str = typer.Option("config.toml")):
    """Watch vault for changes and index continuously."""
    cfg = _cfg(config)
    idx = Indexer(cfg)
    idx.watch()

@app.command()
def mcp(config: str = typer.Option("config.toml")):
    """Run MCP server (stdio)."""
    cfg = _cfg(config)
    run_stdio_server(cfg)

if __name__ == "__main__":
    app()
