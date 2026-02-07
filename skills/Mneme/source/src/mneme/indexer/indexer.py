from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..config import VaultConfig, MultiVaultConfig, VaultDefinition
from .parallel_types import ExtractionResult, ChunkData, ImageTask, ScanStats, ProcessResult, BatchStats
from ..hashing import hash_file, blake2b_hex
from ..paths import relpath
from ..models import Document, Chunk
from ..extractors.base import ExtractorRegistry
from ..extractors.markdown import MarkdownExtractor
from ..extractors.pdf import PdfExtractor
from ..extractors.pptx import PptxExtractor
from ..extractors.xlsx import XlsxExtractor
from ..extractors.image import TesseractImageExtractor, GeminiImageExtractor, GeminiServiceAccountImageExtractor, AIGatewayImageExtractor
from ..chunking.base import ChunkerRegistry
from ..chunking.markdown_chunker import MarkdownChunker
from ..chunking.boundary_chunker import BoundaryMarkerChunker
from ..embeddings.sentence_transformers import SentenceTransformersEmbedder
from ..embeddings.ollama import OllamaEmbedder
from ..store.libsql_store import LibSqlStore
from .queue import JobQueue, Job
from .change_detector import ChangeDetector
from .reconciler import Reconciler

logger = logging.getLogger(__name__)


class BatchCollector:
    """Collects filesystem events into batches for efficient processing.

    Thread-safe accumulator that triggers batch processing when either:
    - Batch reaches max_batch_size files
    - Timeout expires since first job was added

    Usage:
        collector = BatchCollector(max_batch_size=10, batch_timeout_seconds=5.0)

        # In event handler thread:
        batch = collector.add_job(job)
        if batch:
            processor.process_batch(batch)

        # In timeout checker thread:
        batch = collector.flush_if_timeout()
        if batch:
            processor.process_batch(batch)
    """

    def __init__(
        self,
        max_batch_size: int = 10,
        batch_timeout_seconds: float = 5.0,
    ):
        self.max_batch_size = max_batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self._pending_jobs: list[Job] = []
        self._first_job_time: float | None = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def add_job(self, job: Job) -> list[Job] | None:
        """Add a job to the batch.

        Returns the batch if ready (size threshold reached), else None.
        """
        with self._lock:
            self._pending_jobs.append(job)

            # Track when first job was added
            if self._first_job_time is None:
                self._first_job_time = time.time()

            # Check if batch is full
            if len(self._pending_jobs) >= self.max_batch_size:
                return self._flush_locked()

            return None

    def flush_if_timeout(self) -> list[Job] | None:
        """Flush pending jobs if timeout has expired.

        Returns the batch if timeout triggered, else None.
        Should be called periodically from a timer thread.
        """
        with self._lock:
            if not self._pending_jobs:
                return None

            if self._first_job_time is None:
                return None

            elapsed = time.time() - self._first_job_time
            if elapsed >= self.batch_timeout_seconds:
                return self._flush_locked()

            return None

    def flush(self) -> list[Job] | None:
        """Force flush all pending jobs.

        Returns the batch if any jobs pending, else None.
        """
        with self._lock:
            if not self._pending_jobs:
                return None
            return self._flush_locked()

    def _flush_locked(self) -> list[Job]:
        """Flush pending jobs (must hold lock)."""
        batch = self._pending_jobs
        self._pending_jobs = []
        self._first_job_time = None
        self._logger.debug(f"[batch] Flushed batch of {len(batch)} jobs")
        return batch

    def pending_count(self) -> int:
        """Return number of pending jobs."""
        with self._lock:
            return len(self._pending_jobs)


@dataclass
class Indexer:
    cfg: VaultConfig

    def __post_init__(self) -> None:
        self.vault_id = blake2b_hex(str(self.cfg.vault_root).encode("utf-8"))[:12]
        self.store = LibSqlStore(
            self.cfg.index_dir / "vaultrag.sqlite",
            use_faiss=self.cfg.use_faiss,
            faiss_index_type=self.cfg.faiss_index_type,
            faiss_nlist=self.cfg.faiss_nlist,
            faiss_nprobe=self.cfg.faiss_nprobe,
        )
        self.store.init()

        # Extractors
        self.extractors = ExtractorRegistry()
        self.extractors.register(MarkdownExtractor())
        self.extractors.register(PdfExtractor())
        self.extractors.register(PptxExtractor())
        self.extractors.register(XlsxExtractor())
        if self.cfg.image_analysis_provider == "tesseract":
            self.extractors.register(TesseractImageExtractor(ocr_mode="on"))
        elif self.cfg.image_analysis_provider == "gemini":
            self.extractors.register(GeminiImageExtractor(
                api_key=self.cfg.gemini_api_key,
                model=self.cfg.gemini_model,
            ))
        elif self.cfg.image_analysis_provider == "gemini-service-account":
            self.extractors.register(GeminiServiceAccountImageExtractor(
                project_id=self.cfg.gemini_sa_project_id,
                location=self.cfg.gemini_sa_location,
                credentials_file=self.cfg.gemini_sa_credentials_file,
                model=self.cfg.gemini_sa_model,
            ))
        elif self.cfg.image_analysis_provider == "aigateway":
            self.extractors.register(AIGatewayImageExtractor(
                gateway_url=self.cfg.aigateway_url,
                gateway_key=self.cfg.aigateway_key,
                model=self.cfg.aigateway_model,
                timeout=self.cfg.aigateway_timeout,
                endpoint_path=self.cfg.aigateway_endpoint_path,
            ))
        # If "off", no image extractor is registered

        # Chunkers with overlap configuration
        self.chunkers = ChunkerRegistry()
        self.chunkers.register("markdown", MarkdownChunker(
            overlap_chars=self.cfg.overlap_chars,
            max_chunk_size=self.cfg.max_chunk_size,
            preserve_heading_metadata=self.cfg.preserve_heading_metadata,
        ))
        self.chunkers.register("pdf", BoundaryMarkerChunker("PAGE", overlap_chars=self.cfg.overlap_chars))
        self.chunkers.register("pptx", BoundaryMarkerChunker("SLIDE", overlap_chars=self.cfg.overlap_chars))
        self.chunkers.register("xlsx", BoundaryMarkerChunker("SHEET", overlap_chars=self.cfg.overlap_chars))
        self.chunkers.register("image", BoundaryMarkerChunker("IMAGE", overlap_chars=self.cfg.overlap_chars))

        # Embedder with query prefix configuration
        if self.cfg.embedding_provider == "sentence_transformers":
            self.embedder = SentenceTransformersEmbedder(
                model_id=self.cfg.embedding_model,
                model_path=self.cfg.embedding_model_path,
                device=self.cfg.embedding_device,
                batch_size=self.cfg.embedding_batch_size,
                use_query_prefix=self.cfg.use_query_prefix,
                query_prefix=self.cfg.query_prefix,
            )
        elif self.cfg.embedding_provider == "ollama":
            self.embedder = OllamaEmbedder(
                model_id=self.cfg.embedding_model,
                use_query_prefix=self.cfg.use_query_prefix,
                query_prefix=self.cfg.query_prefix,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.cfg.embedding_provider}")

    def scan(self, full: bool = False) -> ScanStats:
        """Scan and index files. `full=True` means re-index all; otherwise only changed.

        Uses parallel scanning if cfg.parallel_scan is True.
        Detects and removes deleted files from the index.
        """
        if self.cfg.parallel_scan:
            return self.scan_parallel(full=full)

        # Sequential scan (original behavior)
        logger = logging.getLogger(__name__)
        start = time.time()
        rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)

        # Get files currently on filesystem
        paths = rec.scan_files()
        fs_files = {relpath(self.cfg.vault_root, p) for p in paths}

        # Get files currently indexed in database
        indexed_files = self.store.get_indexed_files(self.vault_id)

        # Detect deletions: files in DB but not on filesystem
        deleted_files = indexed_files - fs_files
        files_deleted = 0
        for rel_path in deleted_files:
            logger.info(f"Detected deletion: {rel_path}")
            self.store.delete_document(self.vault_id, rel_path)
            files_deleted += 1

        if files_deleted > 0:
            logger.info(f"Removed {files_deleted} deleted file(s) from index")

        # Index existing files
        files_indexed = 0
        for p in paths:
            self._index_one(p, force=full)
            files_indexed += 1

        return ScanStats(
            files_scanned=files_indexed,
            files_indexed=files_indexed,
            files_deleted=files_deleted,
            elapsed_seconds=time.time() - start,
        )

    def scan_parallel(self, full: bool = False) -> ScanStats:
        """Parallel scan with batched embedding and writes.

        Uses the unified _process_file() method for extraction/chunking.

        Phase 0: Detect and remove deleted files from index
        Phase 1: Parallel extraction/chunking via _process_file()
        Phase 2: Batched embedding across files
        Phase 3: Parallel image analysis (if enabled)
        """
        logger = logging.getLogger(__name__)
        start = time.time()

        # Discover files
        rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)
        paths = rec.scan_files()
        logger.info(f"[scan] Found {len(paths)} files to process")

        # Phase 0: Detect and remove deleted files
        fs_files = {relpath(self.cfg.vault_root, p) for p in paths}
        indexed_files = self.store.get_indexed_files(self.vault_id)
        deleted_files = indexed_files - fs_files
        files_deleted = 0
        for rel_path in deleted_files:
            logger.info(f"[scan] Deleted: {rel_path}")
            self.store.delete_document(self.vault_id, rel_path)
            files_deleted += 1

        if files_deleted > 0:
            logger.info(f"[scan] Phase 0: Removed {files_deleted} deleted file(s)")

        # Load manifest for incremental skip (unless full rescan requested)
        if full:
            self._manifest_entries: dict[str, tuple[int, int]] = {}
        else:
            self._manifest_entries = self.store.get_manifest_entries(self.vault_id)
            if self._manifest_entries:
                logger.info(f"[scan] Loaded manifest with {len(self._manifest_entries)} entries for incremental skip")

        # Phase 1: Parallel extraction and chunking using unified _process_file()
        process_results: list[ProcessResult] = []
        image_tasks: list[ImageTask] = []
        files_failed = 0
        files_skipped_unchanged = 0

        with ThreadPoolExecutor(max_workers=self.cfg.extraction_workers) as executor:
            futures = {
                executor.submit(self._process_file, p): p
                for p in paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()

                    # Skip files that should be skipped
                    if result.skipped:
                        if result.skipped_unchanged:
                            files_skipped_unchanged += 1
                        continue

                    # Handle errors
                    if result.error:
                        logger.warning(f"[scan] Failed: {path} - {result.error}")
                        files_failed += 1
                        continue

                    process_results.append(result)

                    # Collect image tasks (already built in ProcessResult)
                    image_tasks.extend(result.image_tasks)

                except Exception as e:
                    logger.error(f"[scan] Worker crashed: {path} - {e}")
                    files_failed += 1

        logger.info(
            f"[scan] Phase 1: {len(process_results)} files extracted, "
            f"{files_skipped_unchanged} unchanged, {files_failed} failed"
        )

        # Phase 2: Batched embedding and storage
        chunks_created, embeddings_created = self._batch_embed_and_store_results(
            process_results
        )
        logger.info(
            f"[scan] Phase 2: {chunks_created} chunks, "
            f"{embeddings_created} embeddings"
        )

        # Phase 3: Parallel image analysis
        images_processed = 0
        if image_tasks and self.cfg.image_analysis_provider != "off":
            images_processed = self._parallel_process_images(image_tasks)
            logger.info(f"[scan] Phase 3: {images_processed} images processed")

        # Save FAISS index at end of scan (ensures final state is persisted)
        self.store.save_faiss_index()

        elapsed = time.time() - start
        logger.info(
            f"[scan] Complete: {len(process_results)} files indexed, "
            f"{files_skipped_unchanged} unchanged in {elapsed:.1f}s"
        )
        return ScanStats(
            files_scanned=len(paths),
            files_indexed=len(process_results),
            files_deleted=files_deleted,
            files_failed=files_failed,
            files_skipped_unchanged=files_skipped_unchanged,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            images_processed=images_processed,
            elapsed_seconds=elapsed,
        )

    def _extract_and_chunk_one(self, abs_path: Path) -> ExtractionResult | None:
        """Extract and chunk a single file (thread-safe, no DB writes).

        Returns ExtractionResult with chunks ready for embedding,
        or None if file should be skipped.
        """
        if not abs_path.is_file():
            return None
        if not abs_path.suffix:
            return None

        extractor = self.extractors.get(abs_path)
        if extractor is None:
            return None

        rel = relpath(self.cfg.vault_root, abs_path)

        try:
            st = abs_path.stat()
            chash = hash_file(abs_path)
            doc_id = blake2b_hex(f"{self.vault_id}:{rel}".encode("utf-8"))[:24]
            file_type = abs_path.suffix.lower().lstrip(".")

            # Extract
            extracted = extractor.extract(abs_path)

            # Determine chunker type label
            if file_type == "md":
                type_label = "markdown"
            elif file_type in ("png", "jpg", "jpeg", "webp"):
                type_label = "image"
            else:
                type_label = file_type

            chunker = self.chunkers.get(type_label)
            if chunker is None:
                return None

            # Filter metadata for chunking
            chunking_metadata = {k: v for k, v in extracted.metadata.items()
                                if k not in ("embedded_images", "image_references", "source_pdf")}

            # Chunk
            chunked = chunker.chunk(extracted.text, chunking_metadata)

            # Build chunk data objects with enriched metadata
            chunks: list[ChunkData] = []

            # Pre-compute enriched metadata fields (shared across all chunks from this file)
            full_path = str(abs_path)
            vault_root = str(self.cfg.vault_root)
            vault_name = self.cfg.vault_name or ""
            file_name = abs_path.name
            file_extension = abs_path.suffix.lower()
            file_size_bytes = int(st.st_size)
            modified_at = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

            # Build Obsidian URI: obsidian://open?vault=<vault_name>&file=<rel_path>
            # Only include if vault_name is set (multi-vault or named single vault)
            obsidian_uri = ""
            if vault_name:
                # URL-encode the path components for Obsidian URI
                encoded_path = quote(rel, safe="")
                obsidian_uri = f"obsidian://open?vault={quote(vault_name, safe='')}&file={encoded_path}"

            for c in chunked:
                text_norm = c.text.strip()
                if not text_norm:
                    continue
                th = blake2b_hex(text_norm.encode("utf-8"))
                cid = blake2b_hex(f"{doc_id}:{c.anchor_type}:{c.anchor_ref}:{th}".encode("utf-8"))[:32]
                meta = dict(c.metadata or {})
                meta.update({
                    # Original fields
                    "rel_path": rel,
                    "file_type": type_label,
                    "anchor_type": c.anchor_type,
                    "anchor_ref": c.anchor_ref,
                    # Enriched metadata for faster operations
                    "full_path": full_path,
                    "vault_root": vault_root,
                    "vault_name": vault_name,
                    "vault_id": self.vault_id,
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "file_size_bytes": file_size_bytes,
                    "modified_at": modified_at,
                    "obsidian_uri": obsidian_uri,
                })
                chunks.append(ChunkData(
                    chunk_id=cid,
                    doc_id=doc_id,
                    vault_id=self.vault_id,
                    anchor_type=c.anchor_type,
                    anchor_ref=c.anchor_ref,
                    text=text_norm,
                    text_hash=th,
                    metadata=meta,
                ))

            # Collect links
            links: list[tuple[str, str]] = []
            if type_label == "markdown":
                for link in extracted.metadata.get("wikilinks", []):
                    links.append((link, "wikilink"))

            return ExtractionResult(
                abs_path=abs_path,
                rel_path=rel,
                doc_id=doc_id,
                vault_id=self.vault_id,
                file_type=type_label,
                content_hash=chash,
                mtime=int(st.st_mtime),
                size=int(st.st_size),
                chunks=chunks,
                embedded_images=extracted.metadata.get("embedded_images", []),
                image_references=extracted.metadata.get("image_references", []),
                links=links,
                error=None,
                # Enriched metadata (for image tasks)
                full_path=full_path,
                vault_root=vault_root,
                vault_name=vault_name,
                file_name=file_name,
                file_extension=file_extension,
                modified_at=modified_at,
                obsidian_uri=obsidian_uri,
            )

        except Exception as e:
            return ExtractionResult(
                abs_path=abs_path,
                rel_path=rel,
                doc_id="",
                vault_id=self.vault_id,
                file_type="",
                content_hash="",
                mtime=0,
                size=0,
                error=str(e),
            )

    def _process_file(self, abs_path: Path) -> ProcessResult:
        """Process a single file: extract, chunk, and prepare for embedding.

        This is the unified processing method used by both scan and watch pipelines.
        Thread-safe, no DB writes - returns ProcessResult with chunks and image tasks.

        Returns:
            ProcessResult with chunks ready for embedding and image tasks for processing.
            On error, returns ProcessResult with error field set.
            If file should be skipped (not a file, no suffix, no extractor), returns
            ProcessResult with skipped=True.
        """
        # Build a minimal result for early returns
        def skipped_result() -> ProcessResult:
            return ProcessResult(
                abs_path=abs_path,
                rel_path="",
                doc_id="",
                vault_id=self.vault_id,
                file_type="",
                content_hash="",
                mtime=0,
                size=0,
                skipped=True,
            )

        def error_result(rel: str, error: str) -> ProcessResult:
            return ProcessResult(
                abs_path=abs_path,
                rel_path=rel,
                doc_id="",
                vault_id=self.vault_id,
                file_type="",
                content_hash="",
                mtime=0,
                size=0,
                error=error,
            )

        # Early validations
        if not abs_path.is_file():
            return skipped_result()
        if not abs_path.suffix:
            return skipped_result()

        extractor = self.extractors.get(abs_path)
        if extractor is None:
            return skipped_result()

        rel = relpath(self.cfg.vault_root, abs_path)

        try:
            st = abs_path.stat()

            # Manifest-based incremental skip: if mtime and size unchanged, skip extraction
            manifest_entries = getattr(self, "_manifest_entries", {})
            if manifest_entries:
                prev = manifest_entries.get(rel)
                if prev is not None:
                    prev_mtime, prev_size = prev
                    if int(st.st_mtime) == prev_mtime and int(st.st_size) == prev_size:
                        result = skipped_result()
                        result.skipped_unchanged = True
                        return result

            chash = hash_file(abs_path)
            doc_id = blake2b_hex(f"{self.vault_id}:{rel}".encode("utf-8"))[:24]
            file_type = abs_path.suffix.lower().lstrip(".")

            # Extract content
            extracted = extractor.extract(abs_path)

            # Determine chunker type label
            if file_type == "md":
                type_label = "markdown"
            elif file_type in ("png", "jpg", "jpeg", "webp"):
                type_label = "image"
            else:
                type_label = file_type

            chunker = self.chunkers.get(type_label)
            if chunker is None:
                return skipped_result()

            # Filter metadata for chunking (exclude image data)
            chunking_metadata = {k: v for k, v in extracted.metadata.items()
                                if k not in ("embedded_images", "image_references", "source_pdf")}

            # Chunk
            chunked = chunker.chunk(extracted.text, chunking_metadata)

            # Pre-compute enriched metadata fields (shared across all chunks from this file)
            full_path = str(abs_path)
            vault_root = str(self.cfg.vault_root)
            vault_name = self.cfg.vault_name or ""
            file_name = abs_path.name
            file_extension = abs_path.suffix.lower()
            file_size_bytes = int(st.st_size)
            modified_at = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

            # Build Obsidian URI
            obsidian_uri = ""
            if vault_name:
                encoded_path = quote(rel, safe="")
                obsidian_uri = f"obsidian://open?vault={quote(vault_name, safe='')}&file={encoded_path}"

            # Build chunk data objects
            chunks: list[ChunkData] = []
            for c in chunked:
                text_norm = c.text.strip()
                if not text_norm:
                    continue
                th = blake2b_hex(text_norm.encode("utf-8"))
                cid = blake2b_hex(f"{doc_id}:{c.anchor_type}:{c.anchor_ref}:{th}".encode("utf-8"))[:32]
                meta = dict(c.metadata or {})
                meta.update({
                    # Original fields
                    "rel_path": rel,
                    "file_type": type_label,
                    "anchor_type": c.anchor_type,
                    "anchor_ref": c.anchor_ref,
                    # Enriched metadata for faster operations
                    "full_path": full_path,
                    "vault_root": vault_root,
                    "vault_name": vault_name,
                    "vault_id": self.vault_id,
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "file_size_bytes": file_size_bytes,
                    "modified_at": modified_at,
                    "obsidian_uri": obsidian_uri,
                })
                chunks.append(ChunkData(
                    chunk_id=cid,
                    doc_id=doc_id,
                    vault_id=self.vault_id,
                    anchor_type=c.anchor_type,
                    anchor_ref=c.anchor_ref,
                    text=text_norm,
                    text_hash=th,
                    metadata=meta,
                ))

            # Collect links
            links: list[tuple[str, str]] = []
            if type_label == "markdown":
                for link in extracted.metadata.get("wikilinks", []):
                    links.append((link, "wikilink"))

            # Build ImageTask objects for embedded images and references
            image_tasks: list[ImageTask] = []
            for img in extracted.metadata.get("embedded_images", []):
                image_tasks.append(ImageTask(
                    parent_doc_id=doc_id,
                    parent_path=rel,
                    vault_id=self.vault_id,
                    file_type=type_label,
                    image_data=img,
                    task_type="embedded",
                    full_path=full_path,
                    vault_root=vault_root,
                    vault_name=vault_name,
                    file_name=file_name,
                    file_extension=file_extension,
                    file_size_bytes=file_size_bytes,
                    modified_at=modified_at,
                    obsidian_uri=obsidian_uri,
                ))
            for ref in extracted.metadata.get("image_references", []):
                image_tasks.append(ImageTask(
                    parent_doc_id=doc_id,
                    parent_path=rel,
                    vault_id=self.vault_id,
                    file_type=type_label,
                    image_data=ref,
                    task_type="reference",
                    full_path=full_path,
                    vault_root=vault_root,
                    vault_name=vault_name,
                    file_name=file_name,
                    file_extension=file_extension,
                    file_size_bytes=file_size_bytes,
                    modified_at=modified_at,
                    obsidian_uri=obsidian_uri,
                ))

            return ProcessResult(
                abs_path=abs_path,
                rel_path=rel,
                doc_id=doc_id,
                vault_id=self.vault_id,
                file_type=type_label,
                content_hash=chash,
                mtime=int(st.st_mtime),
                size=file_size_bytes,
                chunks=chunks,
                image_tasks=image_tasks,
                links=links,
                error=None,
                skipped=False,
                full_path=full_path,
                vault_root=vault_root,
                vault_name=vault_name,
                file_name=file_name,
                file_extension=file_extension,
                modified_at=modified_at,
                obsidian_uri=obsidian_uri,
            )

        except Exception as e:
            return error_result(rel, str(e))

    def _batch_embed_and_store(self, results: list[ExtractionResult]) -> tuple[int, int]:
        """Batch embed chunks across files and write to store.

        Returns (chunks_created, embeddings_created).
        """
        from ..models import Document, Chunk

        # Collect all documents, chunks, and texts
        all_docs: list[Document] = []
        all_chunks: list[Chunk] = []
        all_texts: list[str] = []
        all_chunk_ids: list[str] = []

        # Deduplicate chunks by chunk_id (last writer wins)
        chunk_map: dict[str, tuple[Chunk, str]] = {}

        for result in results:
            doc = Document(
                doc_id=result.doc_id,
                vault_id=result.vault_id,
                rel_path=result.rel_path,
                file_type=result.file_type,
                mtime=result.mtime,
                size=result.size,
                content_hash=result.content_hash,
                deleted=False,
                metadata={"extractor_version": self.cfg.extractor_version},
            )
            all_docs.append(doc)

            for chunk_data in result.chunks:
                chunk = Chunk(
                    chunk_id=chunk_data.chunk_id,
                    doc_id=chunk_data.doc_id,
                    vault_id=chunk_data.vault_id,
                    anchor_type=chunk_data.anchor_type,
                    anchor_ref=chunk_data.anchor_ref,
                    text=chunk_data.text,
                    text_hash=chunk_data.text_hash,
                    metadata=chunk_data.metadata,
                )
                chunk_map[chunk_data.chunk_id] = (chunk, chunk_data.text)

        for chunk, text in chunk_map.values():
            all_chunks.append(chunk)
            all_texts.append(text)
            all_chunk_ids.append(chunk.chunk_id)

        # Batch write documents
        for doc in all_docs:
            self.store.upsert_document(doc)

        # Batch write chunks
        if all_chunks:
            self.store.upsert_chunks(all_chunks)

        # Batch embedding (cross-file) in chunks of embed_batch_size
        embeddings_created = 0
        if all_texts:
            batch_size = self.cfg.embed_batch_size
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_ids = all_chunk_ids[i:i + batch_size]

                vectors = self.embedder.embed_texts(batch_texts)
                self.store.upsert_embeddings(batch_ids, model_id=self.embedder.model_id, vectors=vectors)
                embeddings_created += len(batch_ids)

        return len(all_chunks), embeddings_created

    def _batch_embed_and_store_results(
        self, results: list[ProcessResult]
    ) -> tuple[int, int]:
        """Batch embed chunks from ProcessResult objects and write to store.

        This is the unified batch method used by both scan and watch pipelines.

        Args:
            results: List of ProcessResult from _process_file()

        Returns:
            (chunks_created, embeddings_created)
        """
        from ..models import Document, Chunk

        logger = logging.getLogger(__name__)

        # Collect all documents, chunks, and texts
        all_docs: list[Document] = []
        all_chunks: list[Chunk] = []
        all_texts: list[str] = []
        all_chunk_ids: list[str] = []

        # Use dict to deduplicate chunks by chunk_id (last writer wins,
        # matching ON CONFLICT behavior but avoiding wasted embeddings)
        chunk_map: dict[str, tuple[Chunk, str]] = {}

        for result in results:
            doc = Document(
                doc_id=result.doc_id,
                vault_id=result.vault_id,
                rel_path=result.rel_path,
                file_type=result.file_type,
                mtime=result.mtime,
                size=result.size,
                content_hash=result.content_hash,
                deleted=False,
                metadata={"extractor_version": self.cfg.extractor_version},
            )
            all_docs.append(doc)

            for chunk_data in result.chunks:
                chunk = Chunk(
                    chunk_id=chunk_data.chunk_id,
                    doc_id=chunk_data.doc_id,
                    vault_id=chunk_data.vault_id,
                    anchor_type=chunk_data.anchor_type,
                    anchor_ref=chunk_data.anchor_ref,
                    text=chunk_data.text,
                    text_hash=chunk_data.text_hash,
                    metadata=chunk_data.metadata,
                )
                chunk_map[chunk_data.chunk_id] = (chunk, chunk_data.text)

        # Build deduplicated lists (preserves insertion order in Python 3.7+)
        for chunk, text in chunk_map.values():
            all_chunks.append(chunk)
            all_texts.append(text)
            all_chunk_ids.append(chunk.chunk_id)

        # Compute embeddings OUTSIDE transaction (CPU-bound, no DB lock)
        all_vectors: list[tuple[list[str], np.ndarray]] = []
        embeddings_created = 0
        if all_texts:
            batch_size = self.cfg.embed_batch_size
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_ids = all_chunk_ids[i:i + batch_size]
                vectors = self.embedder.embed_texts(batch_texts)
                all_vectors.append((batch_ids, vectors))
                embeddings_created += len(batch_ids)

        # Write all data in a single transaction (atomic)
        try:
            self.store.begin_transaction()

            for doc in all_docs:
                self.store.upsert_document(doc, _commit=False)

            if all_chunks:
                self.store.upsert_chunks(all_chunks, _commit=False)

            for batch_ids, vectors in all_vectors:
                self.store.upsert_embeddings(
                    batch_ids, model_id=self.embedder.model_id, vectors=vectors, _commit=False
                )

            # Upsert manifest entries for each result
            for result in results:
                self.store.upsert_manifest(
                    vault_id=result.vault_id,
                    rel_path=result.rel_path,
                    file_hash=result.content_hash,
                    chunk_count=len(result.chunks),
                    mtime=result.mtime,
                    size=result.size,
                )

            self.store.commit_transaction()
        except Exception:
            self.store.rollback_transaction()
            raise

        logger.debug(
            f"[batch] Stored {len(all_docs)} docs, {len(all_chunks)} chunks, "
            f"{embeddings_created} embeddings"
        )
        return len(all_chunks), embeddings_created

    def _process_batch(self, jobs: list[Job]) -> BatchStats:
        """Process a batch of jobs using the unified pipeline.

        This is the core batch processing method for watch mode:
        1. Group jobs by type (delete, upsert, move)
        2. Handle deletes first
        3. Parallel extraction via _process_file()
        4. Batch embed and store
        5. Parallel image processing

        Args:
            jobs: List of Job objects from BatchCollector

        Returns:
            BatchStats with processing statistics
        """
        logger = logging.getLogger(__name__)
        start = time.time()

        # Group jobs by type
        delete_jobs: list[Job] = []
        upsert_jobs: list[Job] = []
        move_jobs: list[Job] = []

        for job in jobs:
            if job.kind == "delete":
                delete_jobs.append(job)
            elif job.kind == "upsert":
                upsert_jobs.append(job)
            elif job.kind == "move":
                move_jobs.append(job)

        files_deleted = 0
        files_processed = 0
        files_failed = 0
        chunks_created = 0
        embeddings_created = 0
        images_processed = 0

        # Step 1: Handle deletes first
        for job in delete_jobs:
            try:
                logger.info(f"[watch] Deleted: {job.rel_path}")
                self.store.delete_document(self.vault_id, job.rel_path)
                files_deleted += 1
            except Exception as e:
                logger.warning(f"[watch] Delete failed: {job.rel_path} - {e}")
                files_failed += 1

        # Step 2: Handle moves (delete old, add new to upsert list)
        for job in move_jobs:
            try:
                logger.info(f"[watch] Moved: {job.rel_path} -> {job.new_rel_path}")
                self.store.delete_document(self.vault_id, job.rel_path)
                files_deleted += 1
                # Add new path as upsert
                if job.new_rel_path:
                    upsert_jobs.append(Job(kind="upsert", rel_path=job.new_rel_path))
            except Exception as e:
                logger.warning(f"[watch] Move failed: {job.rel_path} - {e}")
                files_failed += 1

        # Step 3: Parallel extraction for upserts
        if upsert_jobs:
            process_results: list[ProcessResult] = []
            image_tasks: list[ImageTask] = []

            paths = [self.cfg.vault_root / job.rel_path for job in upsert_jobs]

            with ThreadPoolExecutor(max_workers=self.cfg.watch_workers) as executor:
                futures = {
                    executor.submit(self._process_file, p): p
                    for p in paths
                }

                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()

                        if result.skipped:
                            continue

                        if result.error:
                            logger.warning(f"[watch] Failed: {path} - {result.error}")
                            files_failed += 1
                            continue

                        process_results.append(result)
                        image_tasks.extend(result.image_tasks)
                        logger.info(f"[watch] Indexed: {result.rel_path}")

                    except Exception as e:
                        logger.error(f"[watch] Worker crashed: {path} - {e}")
                        files_failed += 1

            files_processed = len(process_results)

            # Step 4: Batch embed and store
            if process_results:
                chunks_created, embeddings_created = self._batch_embed_and_store_results(
                    process_results
                )

            # Step 5: Parallel image processing
            if image_tasks and self.cfg.image_analysis_provider != "off":
                # Use watch-specific image worker count
                original_workers = self.cfg.image_workers
                try:
                    # Temporarily override image workers for watch mode
                    object.__setattr__(self.cfg, 'image_workers', self.cfg.watch_image_workers)
                    images_processed = self._parallel_process_images(image_tasks)
                finally:
                    object.__setattr__(self.cfg, 'image_workers', original_workers)

        elapsed = time.time() - start
        logger.info(
            f"[watch] Batch complete: {files_processed} files, {chunks_created} chunks, "
            f"{files_deleted} deleted in {elapsed:.1f}s"
        )

        return BatchStats(
            files_processed=files_processed,
            files_deleted=files_deleted,
            files_failed=files_failed,
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            images_processed=images_processed,
            elapsed_seconds=elapsed,
        )

    def _parallel_process_images(self, tasks: list[ImageTask]) -> int:
        """Process images in parallel using thread pool.

        Returns number of images successfully processed.
        """
        logger = logging.getLogger(__name__)

        processed = 0
        image_chunks: list[tuple[Chunk, str]] = []  # (chunk, text)

        with ThreadPoolExecutor(max_workers=self.cfg.image_workers) as executor:
            futures = {
                executor.submit(self._process_single_image, task): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result:
                        image_chunks.append(result)
                        processed += 1
                except Exception as e:
                    logger.warning(f"Image processing failed for {task.parent_path}: {e}")

        # Batch embed and store image chunks
        if image_chunks:
            chunks = [c for c, _ in image_chunks]
            texts = [t for _, t in image_chunks]
            ids = [c.chunk_id for c in chunks]

            self.store.upsert_chunks(chunks)

            # Batch embed
            batch_size = self.cfg.embed_batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                vectors = self.embedder.embed_texts(batch_texts)
                self.store.upsert_embeddings(batch_ids, model_id=self.embedder.model_id, vectors=vectors)

        return processed

    def _process_single_image(self, task: ImageTask) -> tuple[Chunk, str] | None:
        """Process a single image (thread-safe, returns chunk data).

        Returns (Chunk, text) or None if processing failed.
        """
        import tempfile
        from ..models import Chunk

        logger = logging.getLogger(__name__)

        # Get image extractor
        image_extractor = None
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            dummy_path = Path(f"dummy{suffix}")
            image_extractor = self.extractors.get(dummy_path)
            if image_extractor is not None:
                break

        if image_extractor is None:
            return None

        tmp_path = None
        try:
            if task.task_type == "embedded":
                # Create temp file for embedded image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                    if "image_bytes" in task.image_data:
                        tmp.write(task.image_data["image_bytes"])
                    else:
                        # PDF case - skip for now (requires fitz which may not be thread-safe)
                        return None

                img_analysis = image_extractor.extract(tmp_path)

                if not img_analysis.text.strip():
                    return None

                # Determine anchor reference
                if "page_num" in task.image_data:
                    anchor_ref = f"page_{task.image_data['page_num']}"
                elif "slide_num" in task.image_data:
                    anchor_ref = f"slide_{task.image_data['slide_num']}"
                else:
                    anchor_ref = "embedded_image"

                anchor_type = f"{task.file_type}_image"

            else:  # reference
                img_path = Path(task.image_data["abs_path"])
                if not img_path.exists():
                    return None

                image_extractor = self.extractors.get(img_path)
                if image_extractor is None:
                    return None

                img_analysis = image_extractor.extract(img_path)

                if not img_analysis.text.strip():
                    return None

                anchor_ref = task.image_data["rel_path"]
                anchor_type = "markdown_image"

            # Create chunk
            text_norm = img_analysis.text.strip()
            th = blake2b_hex(text_norm.encode("utf-8"))
            cid = blake2b_hex(f"{task.parent_doc_id}:{anchor_type}:{anchor_ref}:{th}".encode("utf-8"))[:32]

            meta = {
                # Original fields
                "rel_path": task.parent_path,
                "file_type": task.file_type,
                "anchor_type": anchor_type,
                "anchor_ref": anchor_ref,
                # Enriched metadata for faster operations
                "full_path": task.full_path,
                "vault_root": task.vault_root,
                "vault_name": task.vault_name,
                "vault_id": task.vault_id,
                "file_name": task.file_name,
                "file_extension": task.file_extension,
                "file_size_bytes": task.file_size_bytes,
                "modified_at": task.modified_at,
                "obsidian_uri": task.obsidian_uri,
            }
            if "width" in task.image_data:
                meta["image_width"] = task.image_data["width"]
            if "height" in task.image_data:
                meta["image_height"] = task.image_data["height"]
            meta.update(img_analysis.metadata)

            chunk = Chunk(
                chunk_id=cid,
                doc_id=task.parent_doc_id,
                vault_id=task.vault_id,
                anchor_type=anchor_type,
                anchor_ref=anchor_ref,
                text=text_norm,
                text_hash=th,
                metadata=meta,
            )

            return (chunk, text_norm)

        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return None
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def watch(self) -> None:
        """Watch loop that indexes in response to filesystem events."""
        import threading
        import queue

        q = JobQueue()
        detector = ChangeDetector(
            root=self.cfg.vault_root,
            q=q,
            store=self.store,
            vault_id=self.vault_id,
            ignore=self.cfg.ignore
        )

        # Start detector in background thread
        detector_thread = threading.Thread(target=detector.watch, daemon=True)
        detector_thread.start()

        logger.info(f"Watching {self.cfg.vault_root} for changes. Press Ctrl+C to stop.")

        try:
            # Consume jobs from queue in main thread
            while True:
                try:
                    job = q.get(timeout=1.0)
                except queue.Empty:
                    continue

                try:
                    if job.kind == "upsert":
                        abs_path = self.cfg.vault_root / job.rel_path
                        logger.info(f"[watch] Indexing: {job.rel_path}")
                        self._index_one(abs_path, force=True)

                    elif job.kind == "delete":
                        logger.info(f"[watch] Deleting: {job.rel_path}")
                        self.store.delete_document(self.vault_id, job.rel_path)

                    elif job.kind == "move":
                        # Delete old path
                        logger.info(f"[watch] Moving: {job.rel_path} -> {job.new_rel_path}")
                        self.store.delete_document(self.vault_id, job.rel_path)

                        # Index new path
                        if job.new_rel_path:
                            abs_path = self.cfg.vault_root / job.new_rel_path
                            self._index_one(abs_path, force=True)

                except Exception as e:
                    logger.error(f"[watch] Error processing {job.kind} job for {job.rel_path}: {e}")
                finally:
                    q.task_done()

        except KeyboardInterrupt:
            logger.info("Stopping watch mode...")
            # Give detector thread a moment to clean up
            detector_thread.join(timeout=2.0)

    def watch_batched(self) -> None:
        """Watch loop with batched processing for better efficiency.

        Uses BatchCollector to accumulate filesystem events and processes
        them in batches using parallel extraction and batched embedding.

        Batches are processed when either:
        - Batch reaches watch_batch_size files
        - watch_batch_timeout seconds have elapsed since first job
        """
        import queue as queue_module

        logger = logging.getLogger(__name__)
        q = JobQueue()
        detector = ChangeDetector(
            root=self.cfg.vault_root,
            q=q,
            store=self.store,
            vault_id=self.vault_id,
            ignore=self.cfg.ignore
        )

        # Create batch collector with config values
        collector = BatchCollector(
            max_batch_size=self.cfg.watch_batch_size,
            batch_timeout_seconds=self.cfg.watch_batch_timeout,
        )

        # Start detector in background thread
        detector_thread = threading.Thread(target=detector.watch, daemon=True)
        detector_thread.start()

        # Queue files modified since last index (catch-up on startup)
        detector.queue_stale_files()

        logger.info(f"[watch] Starting batched watch on: {self.cfg.vault_root}")
        logger.info(
            f"[watch] Batch config: size={self.cfg.watch_batch_size}, "
            f"timeout={self.cfg.watch_batch_timeout}s, "
            f"workers={self.cfg.watch_workers}"
        )
        logger.info(f"Watching {self.cfg.vault_root} for changes (batched mode). Press Ctrl+C to stop.")

        def process_batch_if_ready(batch: list[Job] | None) -> None:
            """Process batch if available."""
            if batch:
                self._process_batch(batch)

        try:
            while True:
                try:
                    # Check for new jobs with short timeout
                    job = q.get(timeout=0.5)
                    q.task_done()

                    # Add to collector, process if batch is ready
                    batch = collector.add_job(job)
                    process_batch_if_ready(batch)

                except queue_module.Empty:
                    # No new jobs - check for timeout flush
                    batch = collector.flush_if_timeout()
                    process_batch_if_ready(batch)

        except KeyboardInterrupt:
            # Flush any remaining jobs
            batch = collector.flush()
            if batch:
                logger.info(f"[watch] Flushing {len(batch)} remaining jobs...")
                self._process_batch(batch)

            logger.info("Stopping watch mode...")
            detector_thread.join(timeout=2.0)
            logger.info("[watch] Watch mode stopped")

    def _index_one(self, abs_path: Path, force: bool = False) -> None:
        if not abs_path.is_file():
            return
        if not abs_path.suffix:
            return

        extractor = self.extractors.get(abs_path)
        if extractor is None:
            return

        rel = relpath(self.cfg.vault_root, abs_path)
        st = abs_path.stat()
        chash = hash_file(abs_path)
        doc_id = blake2b_hex(f"{self.vault_id}:{rel}".encode("utf-8"))[:24]
        file_type = abs_path.suffix.lower().lstrip(".")

        extracted = extractor.extract(abs_path)

        # Determine chunker type label
        if file_type == "md":
            type_label = "markdown"
        elif file_type in ("png", "jpg", "jpeg", "webp"):
            type_label = "image"
        else:
            type_label = file_type
        chunker = self.chunkers.get(type_label)
        if chunker is None:
            return

        # Filter out embedded images/references from metadata before chunking
        # (these are processed separately and shouldn't be in every chunk)
        chunking_metadata = {k: v for k, v in extracted.metadata.items()
                            if k not in ("embedded_images", "image_references", "source_pdf")}

        # Chunk
        chunked = chunker.chunk(extracted.text, chunking_metadata)

        doc = Document(
            doc_id=doc_id,
            vault_id=self.vault_id,
            rel_path=rel,
            file_type=type_label,
            mtime=int(st.st_mtime),
            size=int(st.st_size),
            content_hash=chash,
            deleted=False,
            metadata={"extractor_version": self.cfg.extractor_version},
        )
        self.store.upsert_document(doc)

        chunks: list[Chunk] = []
        texts: list[str] = []
        ids: list[str] = []

        # Persist outgoing links (Obsidian-aware)
        if type_label == "markdown":
            links = extracted.metadata.get("wikilinks", [])
            # TODO: store links in `links` table and maintain backlinks

        # Pre-compute enriched metadata fields (shared across all chunks from this file)
        full_path = str(abs_path)
        vault_root = str(self.cfg.vault_root)
        vault_name = self.cfg.vault_name or ""
        file_name = abs_path.name
        file_extension = abs_path.suffix.lower()
        file_size_bytes = int(st.st_size)
        modified_at = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

        # Build Obsidian URI
        obsidian_uri = ""
        if vault_name:
            encoded_path = quote(rel, safe="")
            obsidian_uri = f"obsidian://open?vault={quote(vault_name, safe='')}&file={encoded_path}"

        for c in chunked:
            text_norm = c.text.strip()
            if not text_norm:
                continue
            th = blake2b_hex(text_norm.encode("utf-8"))
            cid = blake2b_hex(f"{doc_id}:{c.anchor_type}:{c.anchor_ref}:{th}".encode("utf-8"))[:32]
            meta = dict(c.metadata or {})
            meta.update({
                # Original fields
                "rel_path": rel,
                "file_type": type_label,
                "anchor_type": c.anchor_type,
                "anchor_ref": c.anchor_ref,
                # Enriched metadata for faster operations
                "full_path": full_path,
                "vault_root": vault_root,
                "vault_name": vault_name,
                "vault_id": self.vault_id,
                "file_name": file_name,
                "file_extension": file_extension,
                "file_size_bytes": file_size_bytes,
                "modified_at": modified_at,
                "obsidian_uri": obsidian_uri,
            })
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                vault_id=self.vault_id,
                anchor_type=c.anchor_type,
                anchor_ref=c.anchor_ref,
                text=text_norm,
                text_hash=th,
                metadata=meta,
            ))
            texts.append(text_norm)
            ids.append(cid)

        self.store.upsert_chunks(chunks)

        # Embed and store vectors
        if ids:
            vecs = self.embedder.embed_texts(texts)
            self.store.upsert_embeddings(ids, model_id=self.embedder.model_id, vectors=vecs)

        # Process embedded images from PDF/PPTX
        if "embedded_images" in extracted.metadata and extracted.metadata["embedded_images"]:
            self._process_embedded_images(
                embedded_images=extracted.metadata["embedded_images"],
                parent_doc_id=doc_id,
                parent_path=rel,
                file_type=type_label,
                source_file=abs_path,
                # Enriched metadata
                full_path=full_path,
                vault_root=vault_root,
                vault_name=vault_name,
                file_name=file_name,
                file_extension=file_extension,
                file_size_bytes=file_size_bytes,
                modified_at=modified_at,
                obsidian_uri=obsidian_uri,
            )

        # Process image references from Markdown
        if "image_references" in extracted.metadata and extracted.metadata["image_references"]:
            self._process_image_references(
                image_refs=extracted.metadata["image_references"],
                parent_doc_id=doc_id,
                parent_path=rel,
                # Enriched metadata
                full_path=full_path,
                vault_root=vault_root,
                vault_name=vault_name,
                file_name=file_name,
                file_extension=file_extension,
                file_size_bytes=file_size_bytes,
                modified_at=modified_at,
                obsidian_uri=obsidian_uri,
            )

        # Write manifest entry for incremental scan compatibility
        self.store.upsert_manifest(
            vault_id=self.vault_id,
            rel_path=rel,
            file_hash=chash,
            chunk_count=len(chunks),
            mtime=int(st.st_mtime),
            size=int(st.st_size),
        )
        self.store._get_conn().commit()

    def _process_embedded_images(
        self,
        embedded_images: list[dict],
        parent_doc_id: str,
        parent_path: str,
        file_type: str,
        source_file: Path,
        # Enriched metadata
        full_path: str = "",
        vault_root: str = "",
        vault_name: str = "",
        file_name: str = "",
        file_extension: str = "",
        file_size_bytes: int = 0,
        modified_at: str = "",
        obsidian_uri: str = "",
    ) -> None:
        """Process images embedded in documents (PDF/PPTX).

        Creates separate chunks for each image linked to parent document.
        Uses temp files to bridge between extraction and analysis.
        """
        import tempfile
        import logging

        logger = logging.getLogger(__name__)

        # Get image extractor (if configured)
        image_extractor = None
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            dummy_path = Path(f"dummy{suffix}")
            image_extractor = self.extractors.get(dummy_path)
            if image_extractor is not None:
                break

        if image_extractor is None:
            logger.debug(f"No image analyzer configured, skipping {len(embedded_images)} embedded images from {parent_path}")
            return

        for idx, img_data in enumerate(embedded_images):
            tmp_path = None
            try:
                # Create temp file for image analysis
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                    if "image_bytes" in img_data:
                        # PPTX case: write bytes directly
                        tmp.write(img_data["image_bytes"])
                    else:
                        # PDF case: extract image from source file using PyMuPDF
                        try:
                            import fitz  # PyMuPDF
                        except ImportError:
                            logger.warning("PyMuPDF not installed, cannot extract PDF images. Install with: pip install pymupdf")
                            return

                        doc = fitz.open(str(source_file))
                        page_num = img_data["page_num"]
                        page = doc[page_num - 1]  # 0-indexed

                        # Get images on page
                        image_list = page.get_images()
                        if not image_list:
                            doc.close()
                            continue

                        # Extract first matching image (simplified approach)
                        # In production, we'd match by bbox coordinates
                        xref = image_list[0][0]  # First image's xref
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        tmp.write(image_bytes)
                        doc.close()

                # Pass through existing image analyzer
                img_analysis = image_extractor.extract(tmp_path)

                # Skip if no text extracted
                if not img_analysis.text.strip():
                    logger.debug(f"No text extracted from embedded image {idx} in {parent_path}")
                    continue

                # Determine anchor reference
                if "page_num" in img_data:
                    anchor_ref = f"page_{img_data['page_num']}"
                elif "slide_num" in img_data:
                    anchor_ref = f"slide_{img_data['slide_num']}"
                else:
                    anchor_ref = f"image_{idx}"

                # Create chunk for image
                text_norm = img_analysis.text.strip()
                th = blake2b_hex(text_norm.encode("utf-8"))
                cid = blake2b_hex(f"{parent_doc_id}:{file_type}_image:{anchor_ref}:{th}".encode("utf-8"))[:32]

                meta = {
                    # Original fields
                    "rel_path": parent_path,
                    "file_type": file_type,
                    "anchor_type": f"{file_type}_image",
                    "anchor_ref": anchor_ref,
                    "image_width": img_data.get("width", 0),
                    "image_height": img_data.get("height", 0),
                    # Enriched metadata for faster operations
                    "full_path": full_path,
                    "vault_root": vault_root,
                    "vault_name": vault_name,
                    "vault_id": self.vault_id,
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "file_size_bytes": file_size_bytes,
                    "modified_at": modified_at,
                    "obsidian_uri": obsidian_uri,
                }
                # Merge in analysis metadata
                meta.update(img_analysis.metadata)

                chunk = Chunk(
                    chunk_id=cid,
                    doc_id=parent_doc_id,
                    vault_id=self.vault_id,
                    anchor_type=f"{file_type}_image",
                    anchor_ref=anchor_ref,
                    text=text_norm,
                    text_hash=th,
                    metadata=meta,
                )

                # Store chunk
                self.store.upsert_chunks([chunk])

                # Embed and store
                embedding = self.embedder.embed_texts([chunk.text])
                self.store.upsert_embeddings([chunk.chunk_id], self.embedder.model_id, embedding)

                logger.debug(f"Indexed embedded image: {anchor_ref} from {parent_path}")

            except Exception as e:
                logger.warning(f"Failed to process embedded image {idx} from {parent_path}: {e}")
            finally:
                # Clean up temp file
                if tmp_path and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

    def _process_image_references(
        self,
        image_refs: list[dict],
        parent_doc_id: str,
        parent_path: str,
        # Enriched metadata
        full_path: str = "",
        vault_root: str = "",
        vault_name: str = "",
        file_name: str = "",
        file_extension: str = "",
        file_size_bytes: int = 0,
        modified_at: str = "",
        obsidian_uri: str = "",
    ) -> None:
        """Process image references from markdown files.

        These are simpler - we already have the file path.
        """
        import logging

        logger = logging.getLogger(__name__)

        for idx, img_ref in enumerate(image_refs):
            try:
                img_path = Path(img_ref["abs_path"])

                if not img_path.exists():
                    logger.debug(f"Image reference not found: {img_ref['rel_path']} from {parent_path}")
                    continue

                # Get image extractor for this image
                image_extractor = self.extractors.get(img_path)
                if image_extractor is None:
                    logger.debug(f"No image analyzer configured for {img_path.suffix}")
                    continue

                # Extract image content
                img_analysis = image_extractor.extract(img_path)

                # Skip if no text extracted
                if not img_analysis.text.strip():
                    logger.debug(f"No text extracted from image reference {img_ref['rel_path']} in {parent_path}")
                    continue

                # Create chunk for referenced image
                text_norm = img_analysis.text.strip()
                th = blake2b_hex(text_norm.encode("utf-8"))
                anchor_ref = img_ref["rel_path"]
                cid = blake2b_hex(f"{parent_doc_id}:markdown_image:{anchor_ref}:{th}".encode("utf-8"))[:32]

                meta = {
                    # Original fields
                    "rel_path": parent_path,
                    "file_type": "markdown",
                    "anchor_type": "markdown_image",
                    "anchor_ref": anchor_ref,
                    "image_path": img_ref["abs_path"],
                    "alt_text": img_ref.get("alt_text", ""),
                    # Enriched metadata for faster operations
                    "full_path": full_path,
                    "vault_root": vault_root,
                    "vault_name": vault_name,
                    "vault_id": self.vault_id,
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "file_size_bytes": file_size_bytes,
                    "modified_at": modified_at,
                    "obsidian_uri": obsidian_uri,
                }
                # Merge in analysis metadata
                meta.update(img_analysis.metadata)

                chunk = Chunk(
                    chunk_id=cid,
                    doc_id=parent_doc_id,
                    vault_id=self.vault_id,
                    anchor_type="markdown_image",
                    anchor_ref=anchor_ref,
                    text=text_norm,
                    text_hash=th,
                    metadata=meta,
                )

                # Store chunk
                self.store.upsert_chunks([chunk])

                # Embed and store
                embedding = self.embedder.embed_texts([chunk.text])
                self.store.upsert_embeddings([chunk.chunk_id], self.embedder.model_id, embedding)

                logger.debug(f"Indexed image reference: {anchor_ref} from {parent_path}")

            except Exception as e:
                logger.warning(f"Failed to process image reference {img_ref.get('rel_path', 'unknown')} from {parent_path}: {e}")


class MultiVaultIndexer:
    """Indexer supporting multiple vaults with a shared database.

    Wraps individual VaultConfig/Indexer instances but uses a shared store
    for cross-vault search capabilities.
    """

    def __init__(self, cfg: MultiVaultConfig) -> None:
        self.cfg = cfg
        self.store = LibSqlStore(
            cfg.index_dir / "vaultrag.sqlite",
            use_faiss=cfg.use_faiss,
            faiss_index_type=cfg.faiss_index_type,
            faiss_nlist=cfg.faiss_nlist,
            faiss_nprobe=cfg.faiss_nprobe,
        )
        self.store.init()

        # Create individual Indexer instances for each vault
        self._indexers: dict[str, Indexer] = {}
        for vault_def in cfg.vaults:
            if not vault_def.enabled:
                continue
            vault_cfg = self._vault_config_from_definition(vault_def)
            indexer = Indexer(vault_cfg)
            # Share the store across all indexers
            indexer.store = self.store
            self._indexers[vault_def.name] = indexer

    def _vault_config_from_definition(self, vault_def: VaultDefinition) -> VaultConfig:
        """Create a VaultConfig from a VaultDefinition using shared settings."""
        # Use vault-specific ignore patterns if provided, otherwise use default
        ignore_patterns = vault_def.ignore if vault_def.ignore else [".git/**", ".obsidian/cache/**", "**/.DS_Store"]

        return VaultConfig(
            vault_root=vault_def.root,
            ignore=ignore_patterns,
            vault_name=vault_def.name,
            index_dir=self.cfg.index_dir,
            extractor_version=self.cfg.extractor_version,
            chunker_version=self.cfg.chunker_version,
            embedding_provider=self.cfg.embedding_provider,
            embedding_model=self.cfg.embedding_model,
            embedding_model_path=self.cfg.embedding_model_path,
            embedding_device=self.cfg.embedding_device,
            embedding_batch_size=self.cfg.embedding_batch_size,
            offline_mode=self.cfg.offline_mode,
            k_vec=self.cfg.k_vec,
            k_lex=self.cfg.k_lex,
            top_k=self.cfg.top_k,
            use_rerank=self.cfg.use_rerank,
            rerank_model=self.cfg.rerank_model,
            rerank_device=self.cfg.rerank_device,
            rerank_top_k=self.cfg.rerank_top_k,
            mcp_transport=self.cfg.mcp_transport,
            image_analysis_provider=self.cfg.image_analysis_provider,
            gemini_api_key=self.cfg.gemini_api_key,
            gemini_model=self.cfg.gemini_model,
            gemini_sa_project_id=self.cfg.gemini_sa_project_id,
            gemini_sa_location=self.cfg.gemini_sa_location,
            gemini_sa_credentials_file=self.cfg.gemini_sa_credentials_file,
            gemini_sa_model=self.cfg.gemini_sa_model,
            extraction_workers=self.cfg.extraction_workers,
            embed_batch_size=self.cfg.embed_batch_size,
            image_workers=self.cfg.image_workers,
            parallel_scan=self.cfg.parallel_scan,
            use_faiss=self.cfg.use_faiss,
            faiss_index_type=self.cfg.faiss_index_type,
            faiss_nlist=self.cfg.faiss_nlist,
            faiss_nprobe=self.cfg.faiss_nprobe,
            overlap_chars=self.cfg.overlap_chars,
            max_chunk_size=self.cfg.max_chunk_size,
            preserve_heading_metadata=self.cfg.preserve_heading_metadata,
            use_query_prefix=self.cfg.use_query_prefix,
            query_prefix=self.cfg.query_prefix,
        )

    def get_vault_names(self) -> list[str]:
        """Get list of configured vault names."""
        return [v.name for v in self.cfg.vaults if v.enabled]

    def get_vault_ids(self, vault_names: list[str] | None = None) -> list[str]:
        """Get vault_ids for specified vault names, or all if None."""
        if vault_names is None:
            return [indexer.vault_id for indexer in self._indexers.values()]

        vault_ids = []
        for name in vault_names:
            if name in self._indexers:
                vault_ids.append(self._indexers[name].vault_id)
        return vault_ids

    def scan(self, full: bool = False, vault_names: list[str] | None = None) -> ScanStats:
        """Scan specified vaults or all if None.

        Args:
            full: Re-index all files if True
            vault_names: List of vault names to scan, or None for all

        Returns:
            Combined ScanStats from all scanned vaults
        """
        logger = logging.getLogger(__name__)

        # Resolve target vaults
        if vault_names is None:
            target_indexers = list(self._indexers.items())
        else:
            target_indexers = [(name, self._indexers[name])
                              for name in vault_names if name in self._indexers]

        if not target_indexers:
            logger.warning("No vaults to scan")
            return ScanStats()

        # Scan each vault and merge stats
        combined_stats = ScanStats()
        for name, indexer in target_indexers:
            logger.info(f"Scanning vault: {name}")
            stats = indexer.scan(full=full)
            combined_stats = self._merge_stats(combined_stats, stats)
            logger.info(f"Vault {name}: {stats.files_indexed} files, {stats.chunks_created} chunks")

        return combined_stats

    def _merge_stats(self, a: ScanStats, b: ScanStats) -> ScanStats:
        """Merge two ScanStats objects."""
        return ScanStats(
            files_scanned=a.files_scanned + b.files_scanned,
            files_indexed=a.files_indexed + b.files_indexed,
            files_deleted=a.files_deleted + b.files_deleted,
            files_failed=a.files_failed + b.files_failed,
            files_skipped_unchanged=a.files_skipped_unchanged + b.files_skipped_unchanged,
            chunks_created=a.chunks_created + b.chunks_created,
            embeddings_created=a.embeddings_created + b.embeddings_created,
            images_processed=a.images_processed + b.images_processed,
            elapsed_seconds=a.elapsed_seconds + b.elapsed_seconds,
        )

    def watch(self, vault_names: list[str] | None = None) -> None:
        """Watch specified vaults for changes.

        Also starts a query server so CLI queries can use the warm model.

        Args:
            vault_names: List of vault names to watch, or None for all
        """
        import threading

        logger = logging.getLogger(__name__)

        # Resolve target vaults
        if vault_names is None:
            target_indexers = list(self._indexers.items())
        else:
            target_indexers = [(name, self._indexers[name])
                              for name in vault_names if name in self._indexers]

        if not target_indexers:
            logger.warning("No vaults to watch")
            return

        logger.info(f"Watching {len(target_indexers)} vault(s). Press Ctrl+C to stop.")
        for name, indexer in target_indexers:
            vault_def = next(v for v in self.cfg.vaults if v.name == name)
            logger.info(f"  - {name}: {vault_def.root}")

        # Start query server with a warm retriever
        query_server = None
        try:
            from ..query_server import QueryServer, get_socket_path
            from ..retrieval.retriever import MultiVaultRetriever

            retriever = MultiVaultRetriever(self.cfg)
            socket_path = get_socket_path(self.cfg.index_dir)
            query_server = QueryServer(retriever, socket_path)
            query_server.start()
        except Exception as e:
            logger.warning(f"Failed to start query server: {e}")

        # Start watch threads for each vault
        threads: list[threading.Thread] = []
        for name, indexer in target_indexers:
            thread = threading.Thread(
                target=self._watch_single_vault,
                args=(name, indexer),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        # Wait for interrupt
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping watch mode...")
        finally:
            if query_server:
                query_server.stop()

    def _watch_single_vault(self, name: str, indexer: Indexer) -> None:
        """Watch a single vault (runs in separate thread)."""
        import queue

        logger = logging.getLogger(__name__)

        from .queue import JobQueue
        from .change_detector import ChangeDetector

        q = JobQueue()
        detector = ChangeDetector(
            root=indexer.cfg.vault_root,
            q=q,
            store=indexer.store,
            vault_id=indexer.vault_id,
            ignore=indexer.cfg.ignore
        )

        # Start detector in background
        import threading
        detector_thread = threading.Thread(target=detector.watch, daemon=True)
        detector_thread.start()

        # Process jobs
        while True:
            try:
                job = q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if job.kind == "upsert":
                    abs_path = indexer.cfg.vault_root / job.rel_path
                    logger.info(f"[{name}] Indexing: {job.rel_path}")
                    indexer._index_one(abs_path, force=True)

                elif job.kind == "delete":
                    logger.info(f"[{name}] Deleting: {job.rel_path}")
                    self.store.delete_document(indexer.vault_id, job.rel_path)

                elif job.kind == "move":
                    logger.info(f"[{name}] Moving: {job.rel_path} -> {job.new_rel_path}")
                    self.store.delete_document(indexer.vault_id, job.rel_path)
                    if job.new_rel_path:
                        abs_path = indexer.cfg.vault_root / job.new_rel_path
                        indexer._index_one(abs_path, force=True)

            except Exception as e:
                logger.error(f"[{name}] Error processing {job.kind} job: {e}")
            finally:
                q.task_done()
