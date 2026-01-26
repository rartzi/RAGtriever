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
    image_analysis_provider: str = "tesseract"  # tesseract|gemini|gemini-service-account|aigateway|off
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Gemini Service Account (for image_analysis_provider="gemini-service-account")
    gemini_sa_project_id: str | None = None
    gemini_sa_location: str = "global"
    gemini_sa_credentials_file: str | None = None
    gemini_sa_model: str = "gemini-2.0-flash-exp"

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
    gemini_sa_timeout: int = 0

    # Retrieval
    k_vec: int = 40
    k_lex: int = 40
    top_k: int = 10
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"  # cpu|cuda|mps
    rerank_top_k: int = 10

    # Fusion algorithm
    fusion_algorithm: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60  # RRF constant (controls rank importance decay)

    # Backlink boost
    backlink_boost_enabled: bool = True
    backlink_boost_weight: float = 0.1  # Score boost per backlink (10%)
    backlink_boost_cap: int = 10  # Maximum backlinks counted for boost

    # Recency boost
    recency_boost_enabled: bool = True
    recency_fresh_days: int = 14  # Files modified within this many days get max boost
    recency_recent_days: int = 60  # Files modified within this many days get medium boost
    recency_old_days: int = 180  # Files older than this get penalty

    # Title/heading boost (DISABLED - content/semantics matter most!)
    heading_boost_enabled: bool = False
    heading_h1_boost: float = 1.05  # 5% boost for H1 (title) chunks
    heading_h2_boost: float = 1.03  # 3% boost for H2 chunks
    heading_h3_boost: float = 1.02  # 2% boost for H3 chunks

    # Tag boost (DISABLED - content/semantics matter most!)
    tag_boost_enabled: bool = False
    tag_boost_weight: float = 0.03  # Score boost per matching tag (3%)
    tag_boost_cap: int = 3  # Maximum tags counted for boost (caps at 9%)

    # Parallelization (scan mode)
    extraction_workers: int = 8       # Number of parallel extraction workers
    embed_batch_size: int = 256       # Cross-file embedding batch size
    image_workers: int = 8            # Number of parallel image API workers
    parallel_scan: bool = True        # Enable parallel scanning

    # Watch mode batching
    watch_workers: int = 4            # Parallel extraction workers for watch
    watch_batch_size: int = 10        # Max files per batch before processing
    watch_batch_timeout: float = 5.0  # Seconds before processing partial batch
    watch_image_workers: int = 4      # Parallel image workers for watch

    # MCP
    mcp_transport: str = "stdio"

    # Logging (audit trail)
    log_dir: str = "logs"                      # Directory for log files (relative to cwd or absolute)
    scan_log_file: str | None = None           # Scan log file path (supports {date}, {datetime})
    watch_log_file: str | None = None          # Watch log file path (supports {date}, {datetime})
    log_level: str = "INFO"                    # Log level: DEBUG, INFO, WARNING, ERROR
    enable_scan_logging: bool = False          # Auto-enable logging for scan command
    enable_watch_logging: bool = False         # Auto-enable logging for watch command

    @staticmethod
    def from_toml(path: str | Path) -> "VaultConfig":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        vault = data.get("vault", {})
        index = data.get("index", {})
        chunking = data.get("chunking", {})
        emb = data.get("embeddings", {})
        img = data.get("image_analysis", {})
        gemini_sa = data.get("gemini_service_account", {})
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

        # Validate fusion algorithm parameters
        fusion_algorithm = ret.get("fusion_algorithm", "rrf")
        if fusion_algorithm not in ("rrf", "weighted"):
            raise ValueError(f"Invalid fusion_algorithm: {fusion_algorithm}. Must be one of: rrf, weighted.")
        rrf_k = int(ret.get("rrf_k", 60))
        if rrf_k < 1 or rrf_k > 1000:
            raise ValueError(f"Invalid rrf_k: {rrf_k}. Must be between 1 and 1000.")

        # Validate backlink boost parameters
        backlink_boost_enabled = bool(ret.get("backlink_boost_enabled", True))
        backlink_boost_weight = float(ret.get("backlink_boost_weight", 0.1))
        if backlink_boost_weight < 0.0 or backlink_boost_weight > 1.0:
            raise ValueError(f"Invalid backlink_boost_weight: {backlink_boost_weight}. Must be between 0.0 and 1.0.")
        backlink_boost_cap = int(ret.get("backlink_boost_cap", 10))
        if backlink_boost_cap < 1 or backlink_boost_cap > 100:
            raise ValueError(f"Invalid backlink_boost_cap: {backlink_boost_cap}. Must be between 1 and 100.")

        # Validate recency boost parameters
        recency_boost_enabled = bool(ret.get("recency_boost_enabled", True))
        recency_fresh_days = int(ret.get("recency_fresh_days", 14))
        recency_recent_days = int(ret.get("recency_recent_days", 60))
        recency_old_days = int(ret.get("recency_old_days", 180))
        if recency_fresh_days < 1 or recency_fresh_days > 365:
            raise ValueError(f"Invalid recency_fresh_days: {recency_fresh_days}. Must be between 1 and 365.")
        if recency_recent_days < 1 or recency_recent_days > 730:
            raise ValueError(f"Invalid recency_recent_days: {recency_recent_days}. Must be between 1 and 730.")
        if recency_old_days < 1 or recency_old_days > 3650:
            raise ValueError(f"Invalid recency_old_days: {recency_old_days}. Must be between 1 and 3650.")
        if recency_fresh_days >= recency_recent_days:
            raise ValueError(f"recency_fresh_days ({recency_fresh_days}) must be less than recency_recent_days ({recency_recent_days}).")
        if recency_recent_days >= recency_old_days:
            raise ValueError(f"recency_recent_days ({recency_recent_days}) must be less than recency_old_days ({recency_old_days}).")

        # Validate and resolve Gemini service account credentials file if specified
        gemini_sa_credentials_file = None
        creds_file = gemini_sa.get("credentials_file")
        if creds_file:
            creds_path = Path(_expand(creds_file)).resolve()
            # Security: Validate file exists and is a regular file
            if not creds_path.exists():
                raise ValueError(f"Gemini service account credentials file not found: {creds_path}")
            if not creds_path.is_file():
                raise ValueError(f"Gemini service account credentials path is not a file: {creds_path}")
            # Security: Check file is readable
            if not os.access(creds_path, os.R_OK):
                raise ValueError(f"Gemini service account credentials file is not readable: {creds_path}")
            gemini_sa_credentials_file = str(creds_path)

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

        # Parse logging configuration
        logging_config = data.get("logging", {})
        log_dir = logging_config.get("dir", "logs")
        scan_log_file = logging_config.get("scan_log_file")
        watch_log_file = logging_config.get("watch_log_file")
        log_level = logging_config.get("level", "INFO").upper()
        enable_scan_logging = bool(logging_config.get("enable_scan_logging", False))
        enable_watch_logging = bool(logging_config.get("enable_watch_logging", False))

        # Validate log level
        valid_log_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {log_level}. Must be one of: {valid_log_levels}")

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
            gemini_sa_project_id=gemini_sa.get("project_id"),
            gemini_sa_location=gemini_sa.get("location", "us-central1"),
            gemini_sa_credentials_file=gemini_sa_credentials_file,
            gemini_sa_model=gemini_sa.get("model", "gemini-2.0-flash-exp"),
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
            gemini_sa_timeout=int(gemini_sa.get("timeout", 0)),
            k_vec=k_vec,
            k_lex=k_lex,
            top_k=top_k,
            use_rerank=bool(ret.get("use_rerank", False)),
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_top_k=rerank_top_k,
            fusion_algorithm=fusion_algorithm,
            rrf_k=rrf_k,
            backlink_boost_enabled=backlink_boost_enabled,
            backlink_boost_weight=backlink_boost_weight,
            backlink_boost_cap=backlink_boost_cap,
            recency_boost_enabled=recency_boost_enabled,
            recency_fresh_days=recency_fresh_days,
            recency_recent_days=recency_recent_days,
            recency_old_days=recency_old_days,
            extraction_workers=int(indexing.get("extraction_workers", 8)),
            embed_batch_size=int(indexing.get("embed_batch_size", 256)),
            image_workers=int(indexing.get("image_workers", 8)),
            parallel_scan=bool(indexing.get("parallel_scan", True)),
            watch_workers=int(indexing.get("watch_workers", 4)),
            watch_batch_size=int(indexing.get("watch_batch_size", 10)),
            watch_batch_timeout=float(indexing.get("watch_batch_timeout", 5.0)),
            watch_image_workers=int(indexing.get("watch_image_workers", 4)),
            mcp_transport=mcp.get("transport", "stdio"),
            log_dir=log_dir,
            scan_log_file=scan_log_file,
            watch_log_file=watch_log_file,
            log_level=log_level,
            enable_scan_logging=enable_scan_logging,
            enable_watch_logging=enable_watch_logging,
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

    # Gemini Service Account
    gemini_sa_project_id: str | None = None
    gemini_sa_location: str = "global"
    gemini_sa_credentials_file: str | None = None
    gemini_sa_model: str = "gemini-2.0-flash-exp"

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
    gemini_sa_timeout: int = 0

    # Retrieval
    k_vec: int = 40
    k_lex: int = 40
    top_k: int = 10
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"
    rerank_top_k: int = 10

    # Fusion algorithm
    fusion_algorithm: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60  # RRF constant (controls rank importance decay)

    # Backlink boost
    backlink_boost_enabled: bool = True
    backlink_boost_weight: float = 0.1  # Score boost per backlink (10%)
    backlink_boost_cap: int = 10  # Maximum backlinks counted for boost

    # Recency boost
    recency_boost_enabled: bool = True
    recency_fresh_days: int = 14  # Files modified within this many days get max boost
    recency_recent_days: int = 60  # Files modified within this many days get medium boost
    recency_old_days: int = 180  # Files older than this get penalty

    # Title/heading boost (DISABLED by default - content/semantics matter most!)
    heading_boost_enabled: bool = False
    heading_h1_boost: float = 1.05  # 5% boost for H1 (title) chunks
    heading_h2_boost: float = 1.03  # 3% boost for H2 chunks
    heading_h3_boost: float = 1.02  # 2% boost for H3 chunks

    # Tag boost (DISABLED by default - content/semantics matter most!)
    tag_boost_enabled: bool = False
    tag_boost_weight: float = 0.03  # Score boost per matching tag (3%)
    tag_boost_cap: int = 3  # Maximum tags counted for boost (caps at 9%)

    # Result diversity (MMR)
    diversity_enabled: bool = True
    max_per_document: int = 2  # Maximum chunks from same document

    # Parallelization (scan mode)
    extraction_workers: int = 8
    embed_batch_size: int = 256
    image_workers: int = 8
    parallel_scan: bool = True

    # Watch mode batching
    watch_workers: int = 4            # Parallel extraction workers for watch
    watch_batch_size: int = 10        # Max files per batch before processing
    watch_batch_timeout: float = 5.0  # Seconds before processing partial batch
    watch_image_workers: int = 4      # Parallel image workers for watch

    # MCP
    mcp_transport: str = "stdio"

    # Logging
    log_dir: str = "logs"                              # Directory for log files
    scan_log_file: str = "logs/scan_{date}.log"        # Scan log file pattern
    watch_log_file: str = "logs/watch_{datetime}.log"  # Watch log file pattern
    log_level: str = "INFO"                            # Log level
    enable_scan_logging: bool = False                  # Auto-enable logging for scan
    enable_watch_logging: bool = True                  # Auto-enable logging for watch

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
        gemini_sa = data.get("gemini_service_account", {})
        aigateway = data.get("aigateway", {})
        ret = data.get("retrieval", {})
        mcp = data.get("mcp", {})
        indexing = data.get("indexing", {})
        logging_config = data.get("logging", {})

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

        # Validate Gemini service account credentials if specified
        gemini_sa_credentials_file = None
        creds_file = gemini_sa.get("credentials_file")
        if creds_file:
            creds_path = Path(_expand(creds_file)).resolve()
            if not creds_path.exists():
                raise ValueError(f"Gemini service account credentials file not found: {creds_path}")
            if not creds_path.is_file():
                raise ValueError(f"Gemini service account credentials path is not a file: {creds_path}")
            if not os.access(creds_path, os.R_OK):
                raise ValueError(f"Gemini service account credentials file is not readable: {creds_path}")
            gemini_sa_credentials_file = str(creds_path)

        # Handle offline mode
        offline_mode_env = os.environ.get("HF_OFFLINE_MODE")
        if offline_mode_env is not None:
            offline_mode = offline_mode_env.lower() in ("1", "true", "yes")
        else:
            offline_mode = emb.get("offline_mode", True)

        if offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Validate fusion algorithm parameters
        fusion_algorithm = ret.get("fusion_algorithm", "rrf")
        if fusion_algorithm not in ("rrf", "weighted"):
            raise ValueError(f"Invalid fusion_algorithm: {fusion_algorithm}. Must be one of: rrf, weighted.")
        rrf_k = int(ret.get("rrf_k", 60))
        if rrf_k < 1 or rrf_k > 1000:
            raise ValueError(f"Invalid rrf_k: {rrf_k}. Must be between 1 and 1000.")

        # Validate backlink boost parameters
        backlink_boost_enabled = bool(ret.get("backlink_boost_enabled", True))
        backlink_boost_weight = float(ret.get("backlink_boost_weight", 0.1))
        if backlink_boost_weight < 0.0 or backlink_boost_weight > 1.0:
            raise ValueError(f"Invalid backlink_boost_weight: {backlink_boost_weight}. Must be between 0.0 and 1.0.")
        backlink_boost_cap = int(ret.get("backlink_boost_cap", 10))
        if backlink_boost_cap < 1 or backlink_boost_cap > 100:
            raise ValueError(f"Invalid backlink_boost_cap: {backlink_boost_cap}. Must be between 1 and 100.")

        # Validate recency boost parameters
        recency_boost_enabled = bool(ret.get("recency_boost_enabled", True))
        recency_fresh_days = int(ret.get("recency_fresh_days", 14))
        recency_recent_days = int(ret.get("recency_recent_days", 60))
        recency_old_days = int(ret.get("recency_old_days", 180))
        if recency_fresh_days < 1 or recency_fresh_days > 365:
            raise ValueError(f"Invalid recency_fresh_days: {recency_fresh_days}. Must be between 1 and 365.")
        if recency_recent_days < 1 or recency_recent_days > 730:
            raise ValueError(f"Invalid recency_recent_days: {recency_recent_days}. Must be between 1 and 730.")
        if recency_old_days < 1 or recency_old_days > 3650:
            raise ValueError(f"Invalid recency_old_days: {recency_old_days}. Must be between 1 and 3650.")
        if recency_fresh_days >= recency_recent_days:
            raise ValueError(f"recency_fresh_days ({recency_fresh_days}) must be less than recency_recent_days ({recency_recent_days}).")
        if recency_recent_days >= recency_old_days:
            raise ValueError(f"recency_recent_days ({recency_recent_days}) must be less than recency_old_days ({recency_old_days}).")

        # Parse heading boost parameters
        heading_boost_enabled = bool(ret.get("heading_boost_enabled", False))
        heading_h1_boost = float(ret.get("heading_h1_boost", 1.05))
        heading_h2_boost = float(ret.get("heading_h2_boost", 1.03))
        heading_h3_boost = float(ret.get("heading_h3_boost", 1.02))

        # Parse tag boost parameters
        tag_boost_enabled = bool(ret.get("tag_boost_enabled", False))
        tag_boost_weight = float(ret.get("tag_boost_weight", 0.03))
        tag_boost_cap = int(ret.get("tag_boost_cap", 3))

        # Parse diversity parameters
        diversity_enabled = bool(ret.get("diversity_enabled", True))
        max_per_document = int(ret.get("max_per_document", 2))

        # Parse logging settings
        log_dir = logging_config.get("dir", "logs")
        scan_log_file = logging_config.get("scan_log_file", "logs/scan_{date}.log")
        watch_log_file = logging_config.get("watch_log_file", "logs/watch_{datetime}.log")
        log_level = logging_config.get("level", "INFO").upper()
        enable_scan_logging = bool(logging_config.get("enable_scan_logging", False))
        enable_watch_logging = bool(logging_config.get("enable_watch_logging", True))

        # Validate log level
        valid_log_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {log_level}. Must be one of: {valid_log_levels}")

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
            gemini_sa_project_id=gemini_sa.get("project_id"),
            gemini_sa_location=gemini_sa.get("location", "us-central1"),
            gemini_sa_credentials_file=gemini_sa_credentials_file,
            gemini_sa_model=gemini_sa.get("model", "gemini-2.0-flash-exp"),
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
            gemini_sa_timeout=int(gemini_sa.get("timeout", 0)),
            k_vec=k_vec,
            k_lex=k_lex,
            top_k=top_k,
            use_rerank=bool(ret.get("use_rerank", False)),
            rerank_model=ret.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            rerank_device=ret.get("rerank_device", "cpu"),
            rerank_top_k=int(ret.get("rerank_top_k", 10)),
            fusion_algorithm=fusion_algorithm,
            rrf_k=rrf_k,
            backlink_boost_enabled=backlink_boost_enabled,
            backlink_boost_weight=backlink_boost_weight,
            backlink_boost_cap=backlink_boost_cap,
            recency_boost_enabled=recency_boost_enabled,
            recency_fresh_days=recency_fresh_days,
            recency_recent_days=recency_recent_days,
            recency_old_days=recency_old_days,
            heading_boost_enabled=heading_boost_enabled,
            heading_h1_boost=heading_h1_boost,
            heading_h2_boost=heading_h2_boost,
            heading_h3_boost=heading_h3_boost,
            tag_boost_enabled=tag_boost_enabled,
            tag_boost_weight=tag_boost_weight,
            tag_boost_cap=tag_boost_cap,
            diversity_enabled=diversity_enabled,
            max_per_document=max_per_document,
            extraction_workers=int(indexing.get("extraction_workers", 8)),
            embed_batch_size=int(indexing.get("embed_batch_size", 256)),
            image_workers=int(indexing.get("image_workers", 8)),
            parallel_scan=bool(indexing.get("parallel_scan", True)),
            watch_workers=int(indexing.get("watch_workers", 4)),
            watch_batch_size=int(indexing.get("watch_batch_size", 10)),
            watch_batch_timeout=float(indexing.get("watch_batch_timeout", 5.0)),
            watch_image_workers=int(indexing.get("watch_image_workers", 4)),
            mcp_transport=mcp.get("transport", "stdio"),
            log_dir=log_dir,
            scan_log_file=scan_log_file,
            watch_log_file=watch_log_file,
            log_level=log_level,
            enable_scan_logging=enable_scan_logging,
            enable_watch_logging=enable_watch_logging,
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