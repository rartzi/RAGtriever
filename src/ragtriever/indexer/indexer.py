from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from ..config import VaultConfig
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

    def scan(self, full: bool = False) -> None:
        """Scan and index files. `full=True` means re-index all; otherwise only changed."""
        rec = Reconciler(self.cfg.vault_root, self.cfg.ignore)
        for p in rec.scan_files():
            self._index_one(p, force=full)

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
