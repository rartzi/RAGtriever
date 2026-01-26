"""Tests for new config fields (chunking and query prefix)."""
import tempfile
from pathlib import Path
import pytest

from mneme.config import VaultConfig


def test_config_chunking_defaults():
    """Test that chunking config has correct defaults."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        # Check chunking defaults
        assert config.overlap_chars == 200
        assert config.max_chunk_size == 2000
        assert config.preserve_heading_metadata is True

        Path(f.name).unlink()


def test_config_chunking_custom():
    """Test custom chunking configuration."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[chunking]
overlap_chars = 100
max_chunk_size = 1500
preserve_heading_metadata = false
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        # Check custom values
        assert config.overlap_chars == 100
        assert config.max_chunk_size == 1500
        assert config.preserve_heading_metadata is False

        Path(f.name).unlink()


def test_config_chunking_validation():
    """Test that invalid chunking values are rejected."""
    # Test invalid overlap_chars
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[chunking]
overlap_chars = 10000
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        with pytest.raises(ValueError, match="Invalid overlap_chars"):
            VaultConfig.from_toml(f.name)

        Path(f.name).unlink()

    # Test invalid max_chunk_size
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[chunking]
max_chunk_size = 100000
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        with pytest.raises(ValueError, match="Invalid max_chunk_size"):
            VaultConfig.from_toml(f.name)

        Path(f.name).unlink()


def test_config_query_prefix_defaults():
    """Test that query prefix config has correct defaults."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        # Check query prefix defaults
        assert config.use_query_prefix is True
        assert config.query_prefix == "Represent this sentence for searching relevant passages: "

        Path(f.name).unlink()


def test_config_query_prefix_custom():
    """Test custom query prefix configuration."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[embeddings]
use_query_prefix = false
query_prefix = "Custom prefix: "
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        # Check custom values
        assert config.use_query_prefix is False
        assert config.query_prefix == "Custom prefix: "

        Path(f.name).unlink()


def test_config_all_new_features():
    """Test config with all new features configured."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[chunking]
overlap_chars = 150
max_chunk_size = 1800
preserve_heading_metadata = true

[embeddings]
provider = "sentence_transformers"
model = "BAAI/bge-small-en-v1.5"
use_query_prefix = true
query_prefix = "Search: "
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        # Check all values
        assert config.overlap_chars == 150
        assert config.max_chunk_size == 1800
        assert config.preserve_heading_metadata is True
        assert config.use_query_prefix is True
        assert config.query_prefix == "Search: "

        Path(f.name).unlink()


def test_config_zero_overlap():
    """Test that overlap can be disabled with 0."""
    toml_content = """
[vault]
root = "/tmp/vault"

[index]
dir = "/tmp/index"

[chunking]
overlap_chars = 0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()

        config = VaultConfig.from_toml(f.name)

        assert config.overlap_chars == 0

        Path(f.name).unlink()
