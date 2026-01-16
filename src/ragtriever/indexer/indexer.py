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
        self.store = LibSqlStore(self.cfg.index_dir / "vaultrag.sqlite")
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

        # Chunk
        chunked = chunker.chunk(extracted.text, extracted.metadata)

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
