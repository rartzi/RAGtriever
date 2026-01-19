from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..config import VaultConfig, MultiVaultConfig, VaultDefinition
from .parallel_types import ExtractionResult, ChunkData, ImageTask, ScanStats
from ..hashing import hash_file, blake2b_hex
from ..paths import relpath
from ..models import Document, Chunk
from ..extractors.base import ExtractorRegistry
from ..extractors.markdown import MarkdownExtractor
from ..extractors.pdf import PdfExtractor
from ..extractors.pptx import PptxExtractor
from ..extractors.xlsx import XlsxExtractor
from ..extractors.image import TesseractImageExtractor, GeminiImageExtractor, VertexAIImageExtractor
from ..chunking.base import ChunkerRegistry
from ..chunking.markdown_chunker import MarkdownChunker
from ..chunking.boundary_chunker import BoundaryMarkerChunker
from ..embeddings.sentence_transformers import SentenceTransformersEmbedder
from ..embeddings.ollama import OllamaEmbedder
from ..store.libsql_store import LibSqlStore
from .queue import JobQueue
from .change_detector import ChangeDetector
from .reconciler import Reconciler

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
        elif self.cfg.image_analysis_provider == "vertex_ai":
            self.extractors.register(VertexAIImageExtractor(
                project_id=self.cfg.vertex_ai_project_id,
                location=self.cfg.vertex_ai_location,
                credentials_file=self.cfg.vertex_ai_credentials_file,
                model=self.cfg.vertex_ai_model,
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
        """
        if self.cfg.parallel_scan:
            return self.scan_parallel(full=full)

        # Sequential scan (original behavior)
        start = time.time()
        rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)
        files_indexed = 0
        for p in rec.scan_files():
            self._index_one(p, force=full)
            files_indexed += 1

        return ScanStats(
            files_scanned=files_indexed,
            files_indexed=files_indexed,
            elapsed_seconds=time.time() - start,
        )

    def scan_parallel(self, full: bool = False) -> ScanStats:
        """Parallel scan with batched embedding and writes.

        Phase 1: Parallel extraction/chunking (ThreadPoolExecutor)
        Phase 2: Batched embedding across files
        Phase 3: Parallel image analysis (if enabled)
        """
        logger = logging.getLogger(__name__)
        start = time.time()

        # Discover files
        rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)
        paths = rec.scan_files()
        logger.info(f"Found {len(paths)} files to index")

        # Phase 1: Parallel extraction and chunking
        extraction_results: list[ExtractionResult] = []
        image_tasks: list[ImageTask] = []
        files_failed = 0

        with ThreadPoolExecutor(max_workers=self.cfg.extraction_workers) as executor:
            futures = {
                executor.submit(self._extract_and_chunk_one, p): p
                for p in paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        continue  # Skipped file

                    if result.error:
                        logger.warning(f"Extraction failed for {path}: {result.error}")
                        files_failed += 1
                        continue

                    extraction_results.append(result)

                    # Collect image tasks for Phase 3
                    for img in result.embedded_images:
                        image_tasks.append(ImageTask(
                            parent_doc_id=result.doc_id,
                            parent_path=result.rel_path,
                            vault_id=result.vault_id,
                            file_type=result.file_type,
                            image_data=img,
                            task_type="embedded",
                        ))
                    for ref in result.image_references:
                        image_tasks.append(ImageTask(
                            parent_doc_id=result.doc_id,
                            parent_path=result.rel_path,
                            vault_id=result.vault_id,
                            file_type=result.file_type,
                            image_data=ref,
                            task_type="reference",
                        ))

                except Exception as e:
                    logger.error(f"Worker crashed for {path}: {e}")
                    files_failed += 1

        logger.info(f"Phase 1 complete: {len(extraction_results)} files extracted, {files_failed} failed")

        # Phase 2: Batched embedding and storage
        chunks_created, embeddings_created = self._batch_embed_and_store(extraction_results)
        logger.info(f"Phase 2 complete: {chunks_created} chunks, {embeddings_created} embeddings")

        # Phase 3: Parallel image analysis
        images_processed = 0
        if image_tasks and self.cfg.image_analysis_provider != "off":
            images_processed = self._parallel_process_images(image_tasks)
            logger.info(f"Phase 3 complete: {images_processed} images processed")

        elapsed = time.time() - start
        return ScanStats(
            files_scanned=len(paths),
            files_indexed=len(extraction_results),
            files_failed=files_failed,
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
                    "rel_path": rel,
                    "file_type": type_label,
                    "anchor_type": c.anchor_type,
                    "anchor_ref": c.anchor_ref,
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
                all_chunks.append(chunk)
                all_texts.append(chunk_data.text)
                all_chunk_ids.append(chunk_data.chunk_id)

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
                "rel_path": task.parent_path,
                "file_type": task.file_type,
                "anchor_type": anchor_type,
                "anchor_ref": anchor_ref,
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
        detector = ChangeDetector(root=self.cfg.vault_root, q=q)

        # Start detector in background thread
        detector_thread = threading.Thread(target=detector.watch, daemon=True)
        detector_thread.start()

        print(f"Watching {self.cfg.vault_root} for changes. Press Ctrl+C to stop.")

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
                        print(f"Indexing: {job.rel_path}")
                        self._index_one(abs_path, force=True)

                    elif job.kind == "delete":
                        print(f"Deleting: {job.rel_path}")
                        self.store.delete_document(self.vault_id, job.rel_path)

                    elif job.kind == "move":
                        # Delete old path
                        print(f"Moving: {job.rel_path} -> {job.new_rel_path}")
                        self.store.delete_document(self.vault_id, job.rel_path)

                        # Index new path
                        if job.new_rel_path:
                            abs_path = self.cfg.vault_root / job.new_rel_path
                            self._index_one(abs_path, force=True)

                except Exception as e:
                    print(f"Error processing {job.kind} job for {job.rel_path}: {e}")
                finally:
                    q.task_done()

        except KeyboardInterrupt:
            print("\nStopping watch mode...")
            # Give detector thread a moment to clean up
            detector_thread.join(timeout=2.0)

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

        # TODO: consult manifest to skip unchanged if not force
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

        for c in chunked:
            text_norm = c.text.strip()
            if not text_norm:
                continue
            th = blake2b_hex(text_norm.encode("utf-8"))
            cid = blake2b_hex(f"{doc_id}:{c.anchor_type}:{c.anchor_ref}:{th}".encode("utf-8"))[:32]
            meta = dict(c.metadata or {})
            meta.update({
                "rel_path": rel,
                "file_type": type_label,
                "anchor_type": c.anchor_type,
                "anchor_ref": c.anchor_ref,
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
            )

        # Process image references from Markdown
        if "image_references" in extracted.metadata and extracted.metadata["image_references"]:
            self._process_image_references(
                image_refs=extracted.metadata["image_references"],
                parent_doc_id=doc_id,
                parent_path=rel,
            )

    def _process_embedded_images(
        self,
        embedded_images: list[dict],
        parent_doc_id: str,
        parent_path: str,
        file_type: str,
        source_file: Path,
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
                    "rel_path": parent_path,
                    "file_type": file_type,
                    "anchor_type": f"{file_type}_image",
                    "anchor_ref": anchor_ref,
                    "image_width": img_data.get("width", 0),
                    "image_height": img_data.get("height", 0),
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
                    "rel_path": parent_path,
                    "file_type": "markdown",
                    "anchor_type": "markdown_image",
                    "anchor_ref": anchor_ref,
                    "image_path": img_ref["abs_path"],
                    "alt_text": img_ref.get("alt_text", ""),
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
            index_dir=self.cfg.index_dir,
            extractor_version=self.cfg.extractor_version,
            chunker_version=self.cfg.chunker_version,
            embedding_provider=self.cfg.embedding_provider,
            embedding_model=self.cfg.embedding_model,
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
            vertex_ai_project_id=self.cfg.vertex_ai_project_id,
            vertex_ai_location=self.cfg.vertex_ai_location,
            vertex_ai_credentials_file=self.cfg.vertex_ai_credentials_file,
            vertex_ai_model=self.cfg.vertex_ai_model,
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
            files_failed=a.files_failed + b.files_failed,
            chunks_created=a.chunks_created + b.chunks_created,
            embeddings_created=a.embeddings_created + b.embeddings_created,
            images_processed=a.images_processed + b.images_processed,
            elapsed_seconds=a.elapsed_seconds + b.elapsed_seconds,
        )

    def watch(self, vault_names: list[str] | None = None) -> None:
        """Watch specified vaults for changes.

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

        print(f"Watching {len(target_indexers)} vault(s). Press Ctrl+C to stop.")
        for name, indexer in target_indexers:
            vault_def = next(v for v in self.cfg.vaults if v.name == name)
            print(f"  - {name}: {vault_def.root}")

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
            print("\nStopping watch mode...")

    def _watch_single_vault(self, name: str, indexer: Indexer) -> None:
        """Watch a single vault (runs in separate thread)."""
        import queue

        logger = logging.getLogger(__name__)

        from .queue import JobQueue
        from .change_detector import ChangeDetector

        q = JobQueue()
        detector = ChangeDetector(root=indexer.cfg.vault_root, q=q)

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
