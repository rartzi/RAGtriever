"""Tests for contextual embeddings (Feature 2: _build_contextual_prefix)."""
from __future__ import annotations

from pathlib import Path

from mneme.indexer.indexer import _build_contextual_prefix
from mneme.config import VaultConfig, MultiVaultConfig


class TestBuildContextualPrefix:
    """Unit tests for _build_contextual_prefix helper."""

    def test_prefix_with_title_and_heading(self):
        """Frontmatter title + heading produces full prefix."""
        meta = {
            "frontmatter": {"title": "My Note"},
            "heading": "Introduction",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: My Note\nSection: Introduction\n\n"

    def test_prefix_filename_fallback(self):
        """No frontmatter title → uses file_name stem."""
        meta = {
            "frontmatter": {},
            "file_name": "meeting-notes.md",
            "heading": "Action Items",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: meeting-notes\nSection: Action Items\n\n"

    def test_prefix_rel_path_fallback(self):
        """No title, no file_name → uses rel_path stem."""
        meta = {
            "rel_path": "projects/roadmap.md",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: roadmap\n\n"

    def test_prefix_no_heading(self):
        """Only Document line when no heading."""
        meta = {
            "frontmatter": {"title": "My Note"},
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: My Note\n\n"

    def test_prefix_empty_metadata(self):
        """Returns empty string when no title can be determined."""
        result = _build_contextual_prefix({})
        assert result == ""

    def test_prefix_frontmatter_not_dict(self):
        """Handles non-dict frontmatter gracefully."""
        meta = {
            "frontmatter": "not-a-dict",
            "file_name": "fallback.md",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: fallback\n\n"

    def test_prefix_empty_title_string(self):
        """Empty title string triggers fallback."""
        meta = {
            "frontmatter": {"title": ""},
            "file_name": "backup.md",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: backup\n\n"

    def test_prefix_heading_empty_string(self):
        """Empty heading string is excluded."""
        meta = {
            "frontmatter": {"title": "My Note"},
            "heading": "",
        }
        result = _build_contextual_prefix(meta)
        assert result == "Document: My Note\n\n"


class TestContextualEmbeddingsConfig:
    """Config wiring for use_contextual_embeddings."""

    def test_disabled_by_default(self, tmp_path: Path):
        """Default config has use_contextual_embeddings=False."""
        cfg = VaultConfig(
            vault_root=tmp_path,
            index_dir=tmp_path / "index",
        )
        assert cfg.use_contextual_embeddings is False

    def test_config_from_toml(self, tmp_path: Path):
        """Parse use_contextual_embeddings from TOML."""
        toml_content = """
[vault]
root = "{root}"

[index]
dir = "{index_dir}"

[embeddings]
use_contextual_embeddings = true
""".format(root=str(tmp_path), index_dir=str(tmp_path / "idx"))

        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_content)

        cfg = VaultConfig.from_toml(toml_path)
        assert cfg.use_contextual_embeddings is True

    def test_multi_vault_config_default(self, tmp_path: Path):
        """MultiVaultConfig defaults to False."""
        toml_content = """
[[vaults]]
name = "test"
root = "{root}"

[index]
dir = "{index_dir}"
""".format(root=str(tmp_path), index_dir=str(tmp_path / "idx"))

        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_content)

        cfg = MultiVaultConfig.from_toml(toml_path)
        assert cfg.use_contextual_embeddings is False

    def test_multi_vault_config_enabled(self, tmp_path: Path):
        """MultiVaultConfig parses use_contextual_embeddings."""
        toml_content = """
[[vaults]]
name = "test"
root = "{root}"

[index]
dir = "{index_dir}"

[embeddings]
use_contextual_embeddings = true
""".format(root=str(tmp_path), index_dir=str(tmp_path / "idx"))

        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_content)

        cfg = MultiVaultConfig.from_toml(toml_path)
        assert cfg.use_contextual_embeddings is True
