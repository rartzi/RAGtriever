from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib

def _expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))


@dataclass
class VaultDefinition:
    """Definition for a single vault in a multi-vault configuration."""

    name: str           # User-friendly name for the vault
    root: Path          # Vault root path
    ignore: list[str] = field(default_factory=list)  # Ignore patterns (optional override)
    enabled: bool = True  # Can disable vaults without removing from config

    def __post_init__(self):
        """Convert string paths to Path objects and expand ~ and environment variables."""
        if isinstance(self.root, str):
            object.__setattr__(self, 'root', Path(_expand(self.root)).resolve())

@dataclass(frozen=True)
class VaultConfig:
    """Configuration for a single vault index.

    This is intentionally minimal; keep config parsing stable.
    """

    vault_root: Path
    index_dir: Path

    ignore: list[str] = field(default_factory=list)
    vault_name: str = ""  # Human-readable vault name (for enriched metadata)

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

    # FAISS (for large-scale vector search)
    use_faiss: bool = False  # Enable for vaults >10K chunks
    faiss_index_type: str = "IVF"  # "Flat" (exact), "IVF" (fast), "HNSW" (fastest)
    faiss_nlist: int = 100  # Number of clusters for IVF
    faiss_nprobe: int = 10  # Number of clusters to search (IVF)

    # Image analysis
    image_analysis_provider: str = "tesseract"  # tesseract|gemini|vertex_ai|aigateway|off
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Vertex AI (for image_analysis_provider="vertex_ai")
    vertex_ai_project_id: str | None = None
    vertex_ai_location: str = "global"
    vertex_ai_credentials_file: str | None = None
    vertex_ai_model: str = "gemini-2.0-flash-exp"

    # Microsoft AI Gateway (for image_analysis_provider="aigateway")
    aigateway_url: str | None = None
    aigateway_key: str | None = None
    aigateway_model: str = "gemini-2.5-flash"
    aigateway_timeout: int = 30000  # Timeout in milliseconds (default, can override)
    aigateway_endpoint_path: str = "vertex-ai-express"  # Path suffix to append to URL

    # Image analysis resilience settings
    image_timeout: int = 30000  # Default timeout for all providers (ms)
    image_max_retries: int = 3  # Number of retries for transient errors
    image_retry_backoff: int = 1000  # Base backoff in ms (doubles each retry)
    image_circuit_threshold: int = 5  # Consecutive failures to trip breaker
    image_circuit_reset: int = 60  # Seconds before breaker auto-resets

    # Per-provider timeout overrides (0 = use image_timeout default)
    gemini_timeout: int = 0
    vertex_ai_timeout: int = 0

    # Retrieval
    k_vec: int = 40
    k_lex: int = 40
    top_k: int = 10
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"  # cpu|cuda|mps
    rerank_top_k: int = 10

    # Parallelization
    extraction_workers: int = 8       # Number of parallel extraction workers
    embed_batch_size: int = 256       # Cross-file embedding batch size
    image_workers: int = 8            # Number of parallel image API workers
    parallel_scan: bool = True        # Enable parallel scanning

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
        aigateway = data.get("aigateway", {})
        ret = data.get("retrieval", {})
        mcp = data.get("mcp", {})
        indexing = data.get("indexing", {})

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

        # Validate reranking parameters
        rerank_model = ret.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        rerank_device = ret.get("rerank_device", "cpu")
        rerank_top_k = int(ret.get("rerank_top_k", 10))

        if rerank_device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Invalid rerank_device: {rerank_device}. Must be one of: cpu, cuda, mps.")
        if rerank_top_k < 1 or rerank_top_k > 1000:
            raise ValueError(f"Invalid rerank_top_k: {rerank_top_k}. Must be between 1 and 1000.")

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

        # Parse and validate FAISS parameters
        use_faiss = bool(emb.get("use_faiss", False))
        faiss_index_type = emb.get("faiss_index_type", "IVF")
        faiss_nlist = int(emb.get("faiss_nlist", 100))
        faiss_nprobe = int(emb.get("faiss_nprobe", 10))

        if faiss_index_type not in ("Flat", "IVF", "HNSW"):
            raise ValueError(f"Invalid faiss_index_type: {faiss_index_type}. Must be one of: Flat, IVF, HNSW.")
        if faiss_nlist < 1 or faiss_nlist > 10000:
            raise ValueError(f"Invalid faiss_nlist: {faiss_nlist}. Must be between 1 and 10000.")
        if faiss_nprobe < 1 or faiss_nprobe > 1000:
            raise ValueError(f"Invalid faiss_nprobe: {faiss_nprobe}. Must be between 1 and 1000.")

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
            use_faiss=use_faiss,
            faiss_index_type=faiss_index_type,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
            image_analysis_provider=img.get("provider", "tesseract"),
            gemini_api_key=img.get("gemini_api_key"),
            gemini_model=img.get("gemini_model", "gemini-2.0-flash"),
            vertex_ai_project_id=vertex.get("project_id"),
            vertex_ai_location=vertex.get("location", "us-central1"),
            vertex_ai_credentials_file=vertex_ai_credentials_file,
            vertex_ai_model=vertex.get("model", "gemini-2.0-flash-exp"),
            aigateway_url=aigateway.get("url") or os.environ.get("AI_GATEWAY_URL"),
            aigateway_key=aigateway.get("key") or os.environ.get("AI_GATEWAY_KEY"),
            aigateway_model=aigateway.get("model", "gemini-2.5-flash"),
            aigateway_timeout=int(aigateway.get("timeout", 0)),  # 0 = use image_timeout
            aigateway_endpoint_path=aigateway.get("endpoint_path", "vertex-ai-express"),
            image_timeout=int(img.get("timeout", 30000)),
            image_max_retries=int(img.get("max_retries", 3)),
            image_retry_backoff=int(img.get("retry_backoff", 1000)),
            image_circuit_threshold=int(img.get("circuit_threshold", 5)),
            image_circuit_reset=int(img.get("circuit_reset", 60)),
            gemini_timeout=int(data.get("gemini", {}).get("timeout", 0)),
            vertex_ai_timeout=int(vertex.get("timeout", 0)),
            k_vec=k_vec,
            k_lex=k_lex,
            top_k=top_k,
            use_rerank=bool(ret.get("use_rerank", False)),
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_top_k=rerank_top_k,
            extraction_workers=int(indexing.get("extraction_workers", 8)),
            embed_batch_size=int(indexing.get("embed_batch_size", 256)),
            image_workers=int(indexing.get("image_workers", 8)),
            parallel_scan=bool(indexing.get("parallel_scan", True)),
            mcp_transport=mcp.get("transport", "stdio"),
        )


@dataclass
class MultiVaultConfig:
    """Configuration for multiple vaults with shared settings.

    All vaults share the same index database (supporting cross-vault search)
    and the same embedding/retrieval settings.
    """

    vaults: list[VaultDefinition]  # List of vault definitions
    index_dir: Path                 # Shared index directory

    # Default ignore patterns (used if vault doesn't specify its own)
    default_ignore: list[str] = field(default_factory=lambda: [
        ".git/**", ".obsidian/cache/**", "**/.DS_Store",
        "**/~$*", "**/.~lock.*"
    ])

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
    embedding_device: str = "cpu"
    offline_mode: bool = True
    use_query_prefix: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # FAISS
    use_faiss: bool = False
    faiss_index_type: str = "IVF"
    faiss_nlist: int = 100
    faiss_nprobe: int = 10

    # Image analysis
    image_analysis_provider: str = "tesseract"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Vertex AI
    vertex_ai_project_id: str | None = None
    vertex_ai_location: str = "global"
    vertex_ai_credentials_file: str | None = None
    vertex_ai_model: str = "gemini-2.0-flash-exp"

    # Microsoft AI Gateway
    aigateway_url: str | None = None
    aigateway_key: str | None = None
    aigateway_model: str = "gemini-2.5-flash"
    aigateway_timeout: int = 30000  # Timeout in milliseconds (default, can override)
    aigateway_endpoint_path: str = "vertex-ai-express"  # Path suffix to append to URL

    # Image analysis resilience settings
    image_timeout: int = 30000  # Default timeout for all providers (ms)
    image_max_retries: int = 3  # Number of retries for transient errors
    image_retry_backoff: int = 1000  # Base backoff in ms (doubles each retry)
    image_circuit_threshold: int = 5  # Consecutive failures to trip breaker
    image_circuit_reset: int = 60  # Seconds before breaker auto-resets

    # Per-provider timeout overrides (0 = use image_timeout default)
    gemini_timeout: int = 0
    vertex_ai_timeout: int = 0

    # Retrieval
    k_vec: int = 40
    k_lex: int = 40
    top_k: int = 10
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"
    rerank_top_k: int = 10

    # Parallelization
    extraction_workers: int = 8
    embed_batch_size: int = 256
    image_workers: int = 8
    parallel_scan: bool = True

    # MCP
    mcp_transport: str = "stdio"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.index_dir, str):
            object.__setattr__(self, 'index_dir', Path(_expand(self.index_dir)).resolve())

    def get_enabled_vaults(self) -> list[VaultDefinition]:
        """Return only enabled vaults."""
        return [v for v in self.vaults if v.enabled]

    def get_vault_by_name(self, name: str) -> VaultDefinition | None:
        """Get a vault by its name."""
        for v in self.vaults:
            if v.name == name:
                return v
        return None

    def get_ignore_patterns(self, vault: VaultDefinition) -> list[str]:
        """Get ignore patterns for a vault (vault-specific or default)."""
        return vault.ignore if vault.ignore else self.default_ignore

    @staticmethod
    def from_toml(path: str | Path) -> "MultiVaultConfig":
        """Parse a multi-vault configuration from TOML file."""
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

        # Parse vault definitions from [[vaults]] array
        vaults_data = data.get("vaults", [])
        if not vaults_data:
            raise ValueError("Multi-vault config requires at least one [[vaults]] section")

        vaults = []
        for v in vaults_data:
            if "name" not in v or "root" not in v:
                raise ValueError("Each vault must have 'name' and 'root' fields")
            vaults.append(VaultDefinition(
                name=v["name"],
                root=Path(_expand(v["root"])).resolve(),
                ignore=list(v.get("ignore", [])),
                enabled=v.get("enabled", True),
            ))

        # Check for duplicate names
        names = [v.name for v in vaults]
        if len(names) != len(set(names)):
            raise ValueError("Vault names must be unique")

        index = data.get("index", {})
        if "dir" not in index:
            raise ValueError("Multi-vault config requires [index] dir setting")
        index_dir = Path(_expand(index["dir"])).resolve()

        # Parse shared settings (same as VaultConfig)
        chunking = data.get("chunking", {})
        emb = data.get("embeddings", {})
        img = data.get("image_analysis", {})
        vertex = data.get("vertex_ai", {})
        aigateway = data.get("aigateway", {})
        ret = data.get("retrieval", {})
        mcp = data.get("mcp", {})
        indexing = data.get("indexing", {})

        # Validate settings (reuse VaultConfig validation logic)
        batch_size = int(emb.get("batch_size", 32))
        if batch_size <= 0 or batch_size > 10000:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be between 1 and 10000.")

        device = emb.get("device", "cpu")
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Invalid device: {device}. Must be one of: cpu, cuda, mps.")

        k_vec = int(ret.get("k_vec", 40))
        k_lex = int(ret.get("k_lex", 40))
        top_k = int(ret.get("top_k", 10))

        # Validate Vertex AI credentials if specified
        vertex_ai_credentials_file = None
        creds_file = vertex.get("credentials_file")
        if creds_file:
            creds_path = Path(_expand(creds_file)).resolve()
            if not creds_path.exists():
                raise ValueError(f"Vertex AI credentials file not found: {creds_path}")
            if not creds_path.is_file():
                raise ValueError(f"Vertex AI credentials path is not a file: {creds_path}")
            if not os.access(creds_path, os.R_OK):
                raise ValueError(f"Vertex AI credentials file is not readable: {creds_path}")
            vertex_ai_credentials_file = str(creds_path)

        # Handle offline mode
        offline_mode_env = os.environ.get("HF_OFFLINE_MODE")
        if offline_mode_env is not None:
            offline_mode = offline_mode_env.lower() in ("1", "true", "yes")
        else:
            offline_mode = emb.get("offline_mode", True)

        if offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        return MultiVaultConfig(
            vaults=vaults,
            index_dir=index_dir,
            default_ignore=list(data.get("default_ignore", [
                ".git/**", ".obsidian/cache/**", "**/.DS_Store",
                "**/~$*", "**/.~lock.*"
            ])),
            extractor_version=index.get("extractor_version", "v1"),
            chunker_version=index.get("chunker_version", "v1"),
            overlap_chars=int(chunking.get("overlap_chars", 200)),
            max_chunk_size=int(chunking.get("max_chunk_size", 2000)),
            preserve_heading_metadata=bool(chunking.get("preserve_heading_metadata", True)),
            embedding_provider=emb.get("provider", "sentence_transformers"),
            embedding_model=emb.get("model", "BAAI/bge-small-en-v1.5"),
            embedding_batch_size=batch_size,
            embedding_device=device,
            offline_mode=offline_mode,
            use_query_prefix=bool(emb.get("use_query_prefix", True)),
            query_prefix=emb.get("query_prefix", "Represent this sentence for searching relevant passages: "),
            use_faiss=bool(emb.get("use_faiss", False)),
            faiss_index_type=emb.get("faiss_index_type", "IVF"),
            faiss_nlist=int(emb.get("faiss_nlist", 100)),
            faiss_nprobe=int(emb.get("faiss_nprobe", 10)),
            image_analysis_provider=img.get("provider", "tesseract"),
            gemini_api_key=img.get("gemini_api_key"),
            gemini_model=img.get("gemini_model", "gemini-2.0-flash"),
            vertex_ai_project_id=vertex.get("project_id"),
            vertex_ai_location=vertex.get("location", "us-central1"),
            vertex_ai_credentials_file=vertex_ai_credentials_file,
            vertex_ai_model=vertex.get("model", "gemini-2.0-flash-exp"),
            aigateway_url=aigateway.get("url") or os.environ.get("AI_GATEWAY_URL"),
            aigateway_key=aigateway.get("key") or os.environ.get("AI_GATEWAY_KEY"),
            aigateway_model=aigateway.get("model", "gemini-2.5-flash"),
            aigateway_timeout=int(aigateway.get("timeout", 0)),  # 0 = use image_timeout
            aigateway_endpoint_path=aigateway.get("endpoint_path", "vertex-ai-express"),
            image_timeout=int(img.get("timeout", 30000)),
            image_max_retries=int(img.get("max_retries", 3)),
            image_retry_backoff=int(img.get("retry_backoff", 1000)),
            image_circuit_threshold=int(img.get("circuit_threshold", 5)),
            image_circuit_reset=int(img.get("circuit_reset", 60)),
            gemini_timeout=int(data.get("gemini", {}).get("timeout", 0)),
            vertex_ai_timeout=int(vertex.get("timeout", 0)),
            k_vec=k_vec,
            k_lex=k_lex,
            top_k=top_k,
            use_rerank=bool(ret.get("use_rerank", False)),
            rerank_model=ret.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            rerank_device=ret.get("rerank_device", "cpu"),
            rerank_top_k=int(ret.get("rerank_top_k", 10)),
            extraction_workers=int(indexing.get("extraction_workers", 8)),
            embed_batch_size=int(indexing.get("embed_batch_size", 256)),
            image_workers=int(indexing.get("image_workers", 8)),
            parallel_scan=bool(indexing.get("parallel_scan", True)),
            mcp_transport=mcp.get("transport", "stdio"),
        )


def load_config(path: str | Path) -> VaultConfig | MultiVaultConfig:
    """Load configuration from TOML file, auto-detecting single vs multi-vault format.

    Returns VaultConfig for single-vault configs (with [vault] section)
    Returns MultiVaultConfig for multi-vault configs (with [[vaults]] array)
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    # Detect config type: [[vaults]] array = multi-vault, [vault] section = single-vault
    if "vaults" in data and isinstance(data["vaults"], list):
        return MultiVaultConfig.from_toml(path)
    elif "vault" in data:
        return VaultConfig.from_toml(path)
    else:
        raise ValueError(
            "Invalid config: must have either [vault] section (single-vault) "
            "or [[vaults]] array (multi-vault)"
        )