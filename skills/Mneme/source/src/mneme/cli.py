from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import typer

from .config import VaultConfig, MultiVaultConfig, load_config
from .indexer.indexer import Indexer, MultiVaultIndexer
from .mcp.server import run_stdio_server
from .retrieval.retriever import Retriever, MultiVaultRetriever

# Suppress harmless multiprocessing resource tracker warnings (common on macOS)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*leaked semaphore")

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _resolve_log_path(log_path_template: str | None) -> str | None:
    """Resolve log file path with date/time pattern substitution.

    Supports:
    - {date}: YYYYMMDD (e.g., 20260123)
    - {datetime}: YYYYMMDD_HHMMSS (e.g., 20260123_142030)
    """
    if not log_path_template:
        return None

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    datetime_str = now.strftime("%Y%m%d_%H%M%S")

    resolved = log_path_template.replace("{date}", date_str).replace("{datetime}", datetime_str)

    # Create parent directory if it doesn't exist
    log_path = Path(resolved)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return str(log_path)

def _cfg(config: str) -> VaultConfig:
    """Load config and ensure it's a single-vault config."""
    cfg = load_config(config)
    if isinstance(cfg, MultiVaultConfig):
        raise typer.BadParameter(
            "This command requires a single-vault config. "
            "Use the multi-vault specific commands or switch to a single [vault] config."
        )
    return cfg

def _multi_cfg(config: str) -> VaultConfig | MultiVaultConfig:
    """Load config, supporting both single and multi-vault formats."""
    return load_config(config)

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
# Set to true to use cached models only (no HuggingFace downloads)
# Can also be controlled via HF_OFFLINE_MODE environment variable
offline_mode = true

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
def scan(
    config: str = typer.Option("config.toml"),
    full: bool = typer.Option(False, help="Re-index all files"),
    parallel: bool = typer.Option(None, help="Override parallel_scan config"),
    workers: int = typer.Option(None, help="Override extraction_workers"),
    vaults: list[str] = typer.Option(None, help="Vault names to scan (multi-vault only, default: all)"),
    log_file: str = typer.Option(None, "--log-file", "-l", help="Log file path for audit trail"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose (DEBUG) logging"),
    profile: str = typer.Option(None, "--profile", "-p", help="Generate profiling report to this file"),
):
    """Scan vault(s) and index. Supports both single-vault and multi-vault configs."""
    import cProfile
    import pstats
    from io import StringIO

    cfg = _multi_cfg(config)

    # Determine log file: CLI flag > config setting > None
    effective_log_file = log_file
    effective_log_level = log_level
    effective_verbose = verbose

    # If no CLI log file but config has logging enabled
    if not effective_log_file and cfg.enable_scan_logging and cfg.scan_log_file:
        effective_log_file = _resolve_log_path(cfg.scan_log_file)
        effective_log_level = cfg.log_level

    # Setup logging if requested
    if effective_log_file or effective_verbose:
        _setup_logging(effective_log_file, effective_log_level, effective_verbose)

    # Wrap scan execution with optional profiling
    profiler = cProfile.Profile() if profile else None

    if profiler:
        profiler.enable()

    if isinstance(cfg, MultiVaultConfig):
        # Multi-vault scan
        if parallel is not None:
            cfg = dataclasses.replace(cfg, parallel_scan=parallel)
        if workers is not None:
            cfg = dataclasses.replace(cfg, extraction_workers=workers)

        idx = MultiVaultIndexer(cfg)
        stats = idx.scan(full=full, vault_names=vaults)

        vault_info = f" across {len(vaults)} vault(s)" if vaults else " across all vaults"
    else:
        # Single-vault scan
        if vaults:
            typer.echo("Warning: --vaults flag ignored for single-vault config", err=True)

        if parallel is not None:
            cfg = dataclasses.replace(cfg, parallel_scan=parallel)
        if workers is not None:
            cfg = dataclasses.replace(cfg, extraction_workers=workers)

        idx = Indexer(cfg)
        stats = idx.scan(full=full)
        vault_info = ""

    if profiler:
        profiler.disable()

        # Write detailed profiling report
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)

        # Sort by cumulative time and print top 50 functions
        ps.strip_dirs().sort_stats("cumulative").print_stats(50)

        # Also print callers for top 20 functions
        s.write("\n\n=== Top 20 Functions by Cumulative Time (with callers) ===\n")
        ps.print_callers(20)

        # Write to file
        Path(profile).write_text(s.getvalue())
        typer.echo(f"Profiling report written to: {profile}")

    if stats.elapsed_seconds > 0:
        typer.echo(f"Scan complete{vault_info}: {stats.files_indexed} files, {stats.chunks_created} chunks in {stats.elapsed_seconds:.1f}s")
        if stats.files_deleted > 0:
            typer.echo(f"  ({stats.files_deleted} deleted files removed from index)")
        if stats.files_failed > 0:
            typer.echo(f"  ({stats.files_failed} files failed)")
        if stats.images_processed > 0:
            typer.echo(f"  ({stats.images_processed} images processed)")
    else:
        typer.echo("Scan complete.")

@app.command()
def query(q: str, config: str = typer.Option("config.toml"), k: int = typer.Option(10),
          path: str = typer.Option("", help="Path prefix filter"),
          rerank: bool = typer.Option(None, help="Override use_rerank config (true/false)"),
          vaults: list[str] = typer.Option(None, help="Vault names to search (multi-vault only, default: all)")):
    """Search the vault(s) with optional reranking. Supports both single-vault and multi-vault configs."""
    cfg = _multi_cfg(config)

    if isinstance(cfg, MultiVaultConfig):
        # Multi-vault query
        if rerank is not None:
            cfg = dataclasses.replace(cfg, use_rerank=rerank)

        r = MultiVaultRetriever(cfg)
        filt = {}
        if path:
            filt["path_prefix"] = path
        hits = r.search(q, k=k, vault_names=vaults, filters=filt)
    else:
        # Single-vault query
        if vaults:
            typer.echo("Warning: --vaults flag ignored for single-vault config", err=True)

        if rerank is not None:
            cfg = dataclasses.replace(cfg, use_rerank=rerank)

        r = Retriever(cfg)
        filt = {}
        if path:
            filt["path_prefix"] = path
        hits = r.search(q, k=k, filters=filt)

    # Enhanced output with reranking info
    results = []
    for h in hits:
        result_dict = {
            "chunk_id": h.chunk_id,
            "score": h.score,
            "snippet": h.snippet,
            "source_ref": h.source_ref.__dict__,
            "metadata": h.metadata,
        }
        # Show reranking info if present
        if h.metadata.get("reranked"):
            result_dict["reranking"] = {
                "original_score": h.metadata.get("original_score"),
                "reranker_score": h.metadata.get("reranker_score"),
                "reranked": True
            }
        results.append(result_dict)

    typer.echo(json.dumps(results, indent=2))

@app.command()
def open(chunk_id: str, config: str = typer.Option("config.toml")):
    """Open a chunk by ID (skeleton)."""
    cfg = _cfg(config)
    r = Retriever(cfg)
    from .models import SourceRef
    sr = SourceRef(vault_id="", rel_path="", file_type="unknown", anchor_type="chunk", anchor_ref=chunk_id, locator={})
    out = r.open(sr)
    typer.echo(out.content)

def _setup_logging(log_file: str | None, log_level: str, verbose: bool) -> None:
    """Configure logging for watch mode."""
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)

    # Format with timestamp for auditability
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    # Always add console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt))
    handlers.append(console)

    # Add rotating file handler if specified
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(file_handler)

    # Configure root logger for mneme
    logger = logging.getLogger("mneme")
    logger.setLevel(level)
    for h in handlers:
        logger.addHandler(h)


@app.command()
def watch(config: str = typer.Option("config.toml"),
          vaults: list[str] = typer.Option(None, help="Vault names to watch (multi-vault only, default: all)"),
          log_file: str = typer.Option(None, "--log-file", "-l", help="Log file path for audit trail"),
          log_level: str = typer.Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
          verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose (DEBUG) logging"),
          batch_size: int = typer.Option(None, "--batch-size", "-b", help="Max files per batch (default: from config)"),
          batch_timeout: float = typer.Option(None, "--batch-timeout", "-t", help="Batch timeout in seconds (default: from config)"),
          no_batch: bool = typer.Option(False, "--no-batch", help="Disable batching, process files one at a time")):
    """Watch vault(s) for changes and index continuously.

    By default, uses batched processing for efficiency. Use --no-batch for legacy serial mode.
    """
    cfg = _multi_cfg(config)

    # Determine log file: CLI flag > config setting > None
    effective_log_file = log_file
    effective_log_level = log_level
    effective_verbose = verbose

    # If no CLI log file but config has logging enabled
    if not effective_log_file and cfg.enable_watch_logging and cfg.watch_log_file:
        effective_log_file = _resolve_log_path(cfg.watch_log_file)
        effective_log_level = cfg.log_level

    # Setup logging before anything else
    _setup_logging(effective_log_file, effective_log_level, effective_verbose)

    # Override batch settings if provided
    if batch_size is not None:
        object.__setattr__(cfg, 'watch_batch_size', batch_size)
    if batch_timeout is not None:
        object.__setattr__(cfg, 'watch_batch_timeout', batch_timeout)

    if isinstance(cfg, MultiVaultConfig):
        idx = MultiVaultIndexer(cfg)
        # Multi-vault indexer uses legacy mode (batching not implemented yet)
        idx.watch(vault_names=vaults)
    else:
        if vaults:
            typer.echo("Warning: --vaults flag ignored for single-vault config", err=True)
        idx = Indexer(cfg)
        if no_batch:
            idx.watch()  # Legacy serial mode
        else:
            idx.watch_batched()  # New batched mode (default)

@app.command(name="list-vaults")
def list_vaults(config: str = typer.Option("config.toml")):
    """List configured vaults (multi-vault config only)."""
    cfg = _multi_cfg(config)

    if isinstance(cfg, MultiVaultConfig):
        typer.echo(f"Configured vaults ({len(cfg.vaults)} total):")
        typer.echo(f"Index directory: {cfg.index_dir}")
        typer.echo("")

        for v in cfg.vaults:
            status = "enabled" if v.enabled else "disabled"
            typer.echo(f"  [{status}] {v.name}")
            typer.echo(f"      Root: {v.root}")
            if v.ignore:
                typer.echo(f"      Ignore patterns: {len(v.ignore)} custom")
            else:
                typer.echo("      Ignore patterns: using defaults")
            typer.echo("")
    else:
        typer.echo("Single-vault configuration detected.")
        typer.echo(f"  Root: {cfg.vault_root}")
        typer.echo(f"  Index: {cfg.index_dir}")

@app.command()
def status(config: str = typer.Option("config.toml"),
           vaults: list[str] = typer.Option(None, help="Vault names to show status for (multi-vault only)")):
    """Show indexing status for vault(s)."""
    cfg = _multi_cfg(config)

    if isinstance(cfg, MultiVaultConfig):
        r = MultiVaultRetriever(cfg)
        status_info = r.status(vault_names=vaults)

        typer.echo(f"Total indexed files: {status_info['total_indexed_files']}")
        typer.echo(f"Total indexed chunks: {status_info['total_indexed_chunks']}")
        typer.echo("")

        for v_status in status_info.get('vaults', []):
            typer.echo(f"  {v_status['name']}:")
            typer.echo(f"    Files: {v_status.get('indexed_files', 0)}")
            typer.echo(f"    Chunks: {v_status.get('indexed_chunks', 0)}")
    else:
        if vaults:
            typer.echo("Warning: --vaults flag ignored for single-vault config", err=True)
        r = Retriever(cfg)
        from .hashing import blake2b_hex
        vault_id = blake2b_hex(str(cfg.vault_root).encode("utf-8"))[:12]
        status_info = r.status(vault_id=vault_id)
        typer.echo(f"Indexed files: {status_info.get('indexed_files', 0)}")
        typer.echo(f"Indexed chunks: {status_info.get('indexed_chunks', 0)}")

@app.command()
def mcp(config: str = typer.Option("config.toml")):
    """Run MCP server (stdio)."""
    cfg = _multi_cfg(config)
    run_stdio_server(cfg)

if __name__ == "__main__":
    app()
