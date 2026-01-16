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

    def __post_init__(self):
        """Convert string paths to Path objects and expand ~ and environment variables."""
        # Convert vault_root to Path if it's a string
        if isinstance(self.vault_root, str):
            object.__setattr__(self, 'vault_root', Path(_expand(self.vault_root)))
        # Convert index_dir to Path if it's a string
        if isinstance(self.index_dir, str):
            object.__setattr__(self, 'index_dir', Path(_expand(self.index_dir)))

    extractor_version: str = "v1"
    chunker_version: str = "v1"

    # Chunking
    overlap_chars: int = 200
    max_chunk_size: int = 2000
    preserve_heading_metadata: bool = True

    # Embeddings
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # cpu|cuda|mps
    offline_mode: bool = True  # Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE
    use_query_prefix: bool = True  # Asymmetric retrieval (BGE pattern)
    query_prefix: str = "Represent this sentence for searching relevant passages: "

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
        chunking = data.get("chunking", {})
        emb = data.get("embeddings", {})
        img = data.get("image_analysis", {})
        vertex = data.get("vertex_ai", {})
        ret = data.get("retrieval", {})
        mcp = data.get("mcp", {})

        vault_root = Path(_expand(vault["root"])).resolve()
        index_dir = Path(_expand(index["dir"])).resolve()

        # Validate and parse batch_size
        batch_size = int(emb.get("batch_size", 32))
        if batch_size <= 0 or batch_size > 10000:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be between 1 and 10000.")

        # Validate device
        device = emb.get("device", "cpu")
        valid_devices = ("cpu", "cuda", "mps")
        if device not in valid_devices:
            raise ValueError(f"Invalid device: {device}. Must be one of {valid_devices}.")

        # Validate retrieval parameters
        k_vec = int(ret.get("k_vec", 40))
        k_lex = int(ret.get("k_lex", 40))
        top_k = int(ret.get("top_k", 10))

        if k_vec <= 0 or k_vec > 1000:
            raise ValueError(f"Invalid k_vec: {k_vec}. Must be between 1 and 1000.")
        if k_lex <= 0 or k_lex > 1000:
            raise ValueError(f"Invalid k_lex: {k_lex}. Must be between 1 and 1000.")
        if top_k <= 0 or top_k > 1000:
            raise ValueError(f"Invalid top_k: {top_k}. Must be between 1 and 1000.")

        # Validate and resolve Vertex AI credentials file if specified
        vertex_ai_credentials_file = None
        creds_file = vertex.get("credentials_file")
        if creds_file:
            creds_path = Path(_expand(creds_file)).resolve()
            # Security: Validate file exists and is a regular file
            if not creds_path.exists():
                raise ValueError(f"Vertex AI credentials file not found: {creds_path}")
            if not creds_path.is_file():
                raise ValueError(f"Vertex AI credentials path is not a file: {creds_path}")
            # Security: Check file is readable
            if not os.access(creds_path, os.R_OK):
                raise ValueError(f"Vertex AI credentials file is not readable: {creds_path}")
            vertex_ai_credentials_file = str(creds_path)

        # Parse offline_mode from config or environment variable
        # Environment variable takes precedence if explicitly set
        offline_mode_env = os.environ.get("HF_OFFLINE_MODE")
        if offline_mode_env is not None:
            offline_mode = offline_mode_env.lower() in ("1", "true", "yes")
        else:
            offline_mode = emb.get("offline_mode", True)

        # SIDE EFFECT: Set HuggingFace offline environment variables based on config
        # This affects the entire Python process. Set during first config load only.
        if offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Parse and validate chunking parameters
        overlap_chars = int(chunking.get("overlap_chars", 200))
        max_chunk_size = int(chunking.get("max_chunk_size", 2000))
        preserve_heading_metadata = bool(chunking.get("preserve_heading_metadata", True))

        if overlap_chars < 0 or overlap_chars > 5000:
            raise ValueError(f"Invalid overlap_chars: {overlap_chars}. Must be between 0 and 5000.")
        if max_chunk_size < 100 or max_chunk_size > 50000:
            raise ValueError(f"Invalid max_chunk_size: {max_chunk_size}. Must be between 100 and 50000.")

        return VaultConfig(
            vault_root=vault_root,
            index_dir=index_dir,
            ignore=list(vault.get("ignore", [])),
            extractor_version=index.get("extractor_version", "v1"),
            chunker_version=index.get("chunker_version", "v1"),
            overlap_chars=overlap_chars,
            max_chunk_size=max_chunk_size,
            preserve_heading_metadata=preserve_heading_metadata,
            embedding_provider=emb.get("provider", "sentence_transformers"),
            embedding_model=emb.get("model", "BAAI/bge-small-en-v1.5"),
            embedding_batch_size=batch_size,
            embedding_device=device,
            offline_mode=offline_mode,
            use_query_prefix=bool(emb.get("use_query_prefix", True)),
            query_prefix=emb.get("query_prefix", "Represent this sentence for searching relevant passages: "),
            image_analysis_provider=img.get("provider", "tesseract"),
            gemini_api_key=img.get("gemini_api_key"),
            gemini_model=img.get("gemini_model", "gemini-2.0-flash"),
            vertex_ai_project_id=vertex.get("project_id"),
            vertex_ai_location=vertex.get("location", "us-central1"),
            vertex_ai_credentials_file=vertex_ai_credentials_file,
            vertex_ai_model=vertex.get("model", "gemini-2.0-flash-exp"),
            k_vec=k_vec,
            k_lex=k_lex,
            top_k=top_k,
            use_rerank=bool(ret.get("use_rerank", False)),
            mcp_transport=mcp.get("transport", "stdio"),
        )