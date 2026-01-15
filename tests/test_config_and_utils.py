"""
Tests for configuration parsing, utilities, and helper functions.
"""

import pytest
from pathlib import Path
import tempfile

from ragtriever.config import VaultConfig
from ragtriever.hashing import blake2b_hex, hash_file
from ragtriever.paths import relpath
from ragtriever.models import Document, Chunk, SearchResult


class TestVaultConfig:
    """Test vault configuration parsing and validation."""

    def test_config_with_basic_paths(self, tmp_path: Path):
        """Test config creation with basic paths."""
        vault = tmp_path / "vault"
        vault.mkdir()
        index = tmp_path / "index"

        config = VaultConfig(
            vault_root=vault,
            index_dir=index,
        )

        assert config.vault_root == vault
        assert config.index_dir == index

    def test_config_defaults(self, tmp_path: Path):
        """Test that config has sensible defaults."""
        vault = tmp_path / "vault"
        vault.mkdir()
        index = tmp_path / "index"

        config = VaultConfig(
            vault_root=vault,
            index_dir=index,
        )

        assert config.embedding_provider is not None
        assert config.embedding_model is not None
        assert config.embedding_device in ["cpu", "cuda", "mps"]

    def test_config_with_custom_embedding_provider(self, tmp_path: Path):
        """Test config with custom embedding provider."""
        vault = tmp_path / "vault"
        vault.mkdir()
        index = tmp_path / "index"

        config = VaultConfig(
            vault_root=vault,
            index_dir=index,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
        )

        assert config.embedding_provider == "ollama"
        assert config.embedding_model == "nomic-embed-text"

    def test_config_image_analysis_modes(self, tmp_path: Path):
        """Test different image analysis provider settings."""
        vault = tmp_path / "vault"
        vault.mkdir()
        index = tmp_path / "index"

        # Off mode
        config1 = VaultConfig(
            vault_root=vault,
            index_dir=index,
            image_analysis_provider="off",
        )
        assert config1.image_analysis_provider == "off"

        # Tesseract mode
        config2 = VaultConfig(
            vault_root=vault,
            index_dir=index,
            image_analysis_provider="tesseract",
        )
        assert config2.image_analysis_provider == "tesseract"

    def test_config_ignore_patterns(self, tmp_path: Path):
        """Test ignore patterns configuration."""
        vault = tmp_path / "vault"
        vault.mkdir()
        index = tmp_path / "index"

        config = VaultConfig(
            vault_root=vault,
            index_dir=index,
            ignore=["*.tmp", ".DS_Store", "node_modules/"],
        )

        assert len(config.ignore) == 3
        assert "*.tmp" in config.ignore


class TestHashing:
    """Test hashing utilities."""

    def test_blake2b_hex_deterministic(self):
        """Test that blake2b_hex produces deterministic output."""
        input_str = "test content"

        hash1 = blake2b_hex(input_str.encode("utf-8"))
        hash2 = blake2b_hex(input_str.encode("utf-8"))

        assert hash1 == hash2

    def test_blake2b_hex_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = blake2b_hex(b"input1")
        hash2 = blake2b_hex(b"input2")

        assert hash1 != hash2

    def test_blake2b_hex_output_format(self):
        """Test hash output format."""
        result = blake2b_hex(b"test")

        # Should be hexadecimal
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)
        # Blake2b produces 64 hex characters for 32 bytes
        assert len(result) == 128

    def test_blake2b_hex_can_truncate(self):
        """Test that hash can be truncated."""
        full_hash = blake2b_hex(b"test")
        truncated = full_hash[:12]

        assert len(truncated) == 12
        assert all(c in "0123456789abcdef" for c in truncated)

    def test_hash_file_deterministic(self, tmp_path: Path):
        """Test that file hashing is deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")

        hash1 = hash_file(test_file)
        hash2 = hash_file(test_file)

        assert hash1 == hash2

    def test_hash_file_detects_changes(self, tmp_path: Path):
        """Test that file hash changes when content changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        hash1 = hash_file(test_file)

        test_file.write_text("modified content")
        hash2 = hash_file(test_file)

        assert hash1 != hash2

    def test_hash_file_binary(self, tmp_path: Path):
        """Test hashing binary files."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")

        result = hash_file(test_file)

        assert isinstance(result, str)
        assert len(result) > 0


class TestPathUtilities:
    """Test path manipulation utilities."""

    def test_relpath_basic(self, tmp_path: Path):
        """Test relative path calculation."""
        root = tmp_path / "root"
        root.mkdir()
        file_path = root / "subdir" / "file.txt"
        file_path.parent.mkdir()
        file_path.write_text("content")

        rel = relpath(root, file_path)

        assert str(rel) == "subdir/file.txt"

    def test_relpath_with_dots(self, tmp_path: Path):
        """Test relpath doesn't include dot components."""
        root = tmp_path / "root"
        root.mkdir()
        file_path = root / "file.txt"
        file_path.write_text("content")

        rel = relpath(root, file_path)

        assert ".." not in str(rel)
        assert str(rel) == "file.txt"

    def test_relpath_nested(self, tmp_path: Path):
        """Test relpath with nested directories."""
        root = tmp_path / "root"
        root.mkdir()
        file_path = root / "a" / "b" / "c" / "file.txt"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("content")

        rel = relpath(root, file_path)

        assert str(rel) == "a/b/c/file.txt"


class TestDataModels:
    """Test data model classes."""

    def test_document_creation(self):
        """Test Document creation."""
        doc = Document(
            doc_id="doc123",
            vault_id="vault456",
            rel_path="path/to/file.md",
            file_type="markdown",
            mtime=1234567890,
            size=1024,
            content_hash="hash123",
            deleted=False,
            metadata={"key": "value"},
        )

        assert doc.doc_id == "doc123"
        assert doc.vault_id == "vault456"
        assert doc.rel_path == "path/to/file.md"
        assert doc.file_type == "markdown"

    def test_chunk_creation(self):
        """Test Chunk creation."""
        chunk = Chunk(
            chunk_id="chunk123",
            doc_id="doc456",
            vault_id="vault789",
            anchor_type="heading",
            anchor_ref="section-1",
            text="Chunk text content",
            text_hash="texthash123",
            metadata={"level": 2},
        )

        assert chunk.chunk_id == "chunk123"
        assert chunk.doc_id == "doc456"
        assert chunk.text == "Chunk text content"
        assert chunk.anchor_type == "heading"

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            vault_id="v1",
            anchor_type="heading",
            anchor_ref="ref",
            text="text",
            text_hash="hash",
        )

        result = SearchResult(
            chunk=chunk,
            score=0.95,
            context={"before": "context before", "after": "context after"},
        )

        assert result.chunk == chunk
        assert result.score == 0.95
        assert "before" in result.context

    def test_document_with_no_metadata(self):
        """Test Document without metadata."""
        doc = Document(
            doc_id="doc1",
            vault_id="vault1",
            rel_path="file.md",
            file_type="markdown",
            mtime=0,
            size=0,
            content_hash="hash",
            deleted=False,
        )

        assert doc.metadata is None or isinstance(doc.metadata, dict)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_safe_read_text_valid_file(self, tmp_path: Path):
        """Test reading valid file."""
        from ragtriever.utils import safe_read_text

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = safe_read_text(test_file)

        assert result == "test content"

    def test_safe_read_text_missing_file(self):
        """Test reading missing file returns None."""
        from ragtriever.utils import safe_read_text

        result = safe_read_text(Path("/nonexistent/file.txt"))

        assert result is None

    def test_safe_read_text_encoding(self, tmp_path: Path):
        """Test reading file with different encoding."""
        from ragtriever.utils import safe_read_text

        test_file = tmp_path / "test.txt"
        test_content = "UTF-8 content: café, naïve"
        test_file.write_text(test_content, encoding="utf-8")

        result = safe_read_text(test_file)

        assert result == test_content


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_requires_vault_root(self):
        """Test that vault_root is required."""
        with pytest.raises((TypeError, ValueError)):
            VaultConfig(index_dir=Path("/tmp"))

    def test_config_requires_index_dir(self):
        """Test that index_dir is required."""
        with pytest.raises((TypeError, ValueError)):
            VaultConfig(vault_root=Path("/tmp"))

    def test_config_accepts_string_paths(self, tmp_path: Path):
        """Test that config accepts string paths."""
        vault = str(tmp_path / "vault")
        index = str(tmp_path / "index")

        config = VaultConfig(
            vault_root=vault,
            index_dir=index,
        )

        # Should convert to Path
        assert isinstance(config.vault_root, Path)
        assert isinstance(config.index_dir, Path)


class TestEnvironmentVariables:
    """Test environment variable handling in config."""

    def test_config_with_home_expansion(self, tmp_path: Path, monkeypatch):
        """Test that ~ is expanded in paths."""
        import os

        # Set HOME to tmp for testing
        monkeypatch.setenv("HOME", str(tmp_path))

        vault_path = "~/vault"
        index_path = "~/index"

        config = VaultConfig(
            vault_root=vault_path,
            index_dir=index_path,
        )

        # Should expand ~
        assert "~" not in str(config.vault_root)
        assert "~" not in str(config.index_dir)


class TestHashingConsistency:
    """Test hash consistency across operations."""

    def test_vault_id_consistent(self, tmp_path: Path):
        """Test that vault ID is consistent for same vault."""
        vault = tmp_path / "vault"
        vault.mkdir()

        from ragtriever.hashing import blake2b_hex

        vault_id1 = blake2b_hex(str(vault).encode("utf-8"))[:12]
        vault_id2 = blake2b_hex(str(vault).encode("utf-8"))[:12]

        assert vault_id1 == vault_id2

    def test_doc_id_deterministic(self, tmp_path: Path):
        """Test that document IDs are deterministic."""
        from ragtriever.hashing import blake2b_hex

        vault_id = "vault123"
        rel_path = "path/to/file.md"

        doc_id1 = blake2b_hex(f"{vault_id}:{rel_path}".encode("utf-8"))[:24]
        doc_id2 = blake2b_hex(f"{vault_id}:{rel_path}".encode("utf-8"))[:24]

        assert doc_id1 == doc_id2

    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic."""
        from ragtriever.hashing import blake2b_hex

        doc_id = "doc123"
        anchor_type = "heading"
        anchor_ref = "section-1"
        text = "chunk text"
        text_hash = blake2b_hex(text.encode("utf-8"))

        chunk_id1 = blake2b_hex(
            f"{doc_id}:{anchor_type}:{anchor_ref}:{text_hash}".encode("utf-8")
        )[:32]
        chunk_id2 = blake2b_hex(
            f"{doc_id}:{anchor_type}:{anchor_ref}:{text_hash}".encode("utf-8")
        )[:32]

        assert chunk_id1 == chunk_id2
