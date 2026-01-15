from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib
from typing import Any

def _expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))

@dataclass(frozen=True)
class VaultConfig:
    """Configuration for a single vault index.

    This is intentionally minimal; keep config parsing stable.
    """

    vault_root: Path
    index_dir: Path

    ignore: list[str] = field(default_factory=list)

    extractor_version: str = "v1"
    chunker_version: str = "v1"

    # Embeddings
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # cpu|cuda|mps

    # Image analysis
    image_analysis_provider: str = "tesseract"  # tesseract|gemini|vertex_ai|off
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Vertex AI (for image_analysis_provider="vertex_ai")
    vertex_ai_project_id: str | None = None
    vertex_ai_location: str = "global"
    vertex_ai_credentials_file: str | None = None
    vertex_ai_model: str = "gemini-2.0-flash-exp"

    # Retrieval
    k_vec: int = 40
    k_lex: int = 40
    top_k: int = 10
    use_rerank: bool = False

    # MCP
    mcp_transport: str = "stdio"

    @staticmethod
    def from_toml(path: str | Path) -> "VaultConfig":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        vault = data.get("vault", {})
        index = data.get("index", {})
        emb = data.get("embeddings", {})
        img = data.get("image_analysis", {})
        vertex = data.get("vertex_ai", {})
        ret = data.get("retrieval", {})
        mcp = data.get("mcp", {})

        vault_root = Path(_expand(vault["root"])).resolve()
        index_dir = Path(_expand(index["dir"])).resolve()

        return VaultConfig(
            vault_root=vault_root,
            index_dir=index_dir,
            ignore=list(vault.get("ignore", [])),
            extractor_version=index.get("extractor_version", "v1"),
            chunker_version=index.get("chunker_version", "v1"),
            embedding_provider=emb.get("provider", "sentence_transformers"),
            embedding_model=emb.get("model", "BAAI/bge-small-en-v1.5"),
            embedding_batch_size=int(emb.get("batch_size", 32)),
            embedding_device=emb.get("device", "cpu"),
            image_analysis_provider=img.get("provider", "tesseract"),
            gemini_api_key=img.get("gemini_api_key"),
            gemini_model=img.get("gemini_model", "gemini-2.0-flash"),
            vertex_ai_project_id=vertex.get("project_id"),
            vertex_ai_location=vertex.get("location", "us-central1"),
            vertex_ai_credentials_file=vertex.get("credentials_file"),
            vertex_ai_model=vertex.get("model", "gemini-2.0-flash-exp"),
            k_vec=int(ret.get("k_vec", 40)),
            k_lex=int(ret.get("k_lex", 40)),
            top_k=int(ret.get("top_k", 10)),
            use_rerank=bool(ret.get("use_rerank", False)),
            mcp_transport=mcp.get("transport", "stdio"),
        )