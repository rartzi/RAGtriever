"""Tests for chunk overlap functionality."""
from src.ragtriever.chunking.markdown_chunker import MarkdownChunker
from src.ragtriever.chunking.boundary_chunker import BoundaryMarkerChunker


def test_markdown_overlap_basic():
    """Test basic overlap between markdown sections."""
    text = """# Section 1
This is content for section one. It has some important context at the end.

# Section 2
This is content for section two.

# Section 3
This is content for section three."""

    chunker = MarkdownChunker(overlap_chars=50, max_chunk_size=2000)
    chunks = chunker.chunk(text, {})

    assert len(chunks) == 3

    # Second chunk should have overlap from first
    assert chunks[1].metadata.get("has_prefix_overlap") is True
    # Third chunk should have overlap from second
    assert chunks[2].metadata.get("has_prefix_overlap") is True

    # First chunk has no overlap
    assert chunks[0].metadata.get("has_prefix_overlap") is None


def test_markdown_no_overlap():
    """Test that overlap can be disabled."""
    text = """# Section 1
Content one.

# Section 2
Content two."""

    chunker = MarkdownChunker(overlap_chars=0, max_chunk_size=2000)
    chunks = chunker.chunk(text, {})

    assert len(chunks) == 2
    # No chunks should have overlap
    assert all(c.metadata.get("has_prefix_overlap") is None for c in chunks)


def test_markdown_large_section_split():
    """Test splitting of large sections."""
    # Create a large section that exceeds max_chunk_size
    large_content = "A" * 3000
    text = f"# Large Section\n{large_content}"

    chunker = MarkdownChunker(overlap_chars=200, max_chunk_size=2000)
    chunks = chunker.chunk(text, {})

    # Should be split into multiple chunks
    assert len(chunks) > 1

    # All chunks should have same heading
    assert all(c.metadata.get("heading") == "Large Section" for c in chunks)

    # All chunks should be marked as splits
    assert all(c.metadata.get("is_split") is True for c in chunks)

    # Should have split_index and split_total
    assert chunks[0].metadata.get("split_index") == 0
    assert chunks[-1].metadata.get("split_index") == len(chunks) - 1
    assert all(c.metadata.get("split_total") == len(chunks) for c in chunks)


def test_markdown_preserve_heading_metadata():
    """Test that heading metadata is preserved."""
    text = """# Top Level
Content here.

## Second Level
More content.

### Third Level
Even more content."""

    chunker = MarkdownChunker(overlap_chars=50, max_chunk_size=2000, preserve_heading_metadata=True)
    chunks = chunker.chunk(text, {})

    # Check that heading and level are preserved
    assert chunks[0].metadata["heading"] == "Top Level"
    assert chunks[0].metadata["level"] == 1

    assert chunks[1].metadata["heading"] == "Second Level"
    assert chunks[1].metadata["level"] == 2

    assert chunks[2].metadata["heading"] == "Third Level"
    assert chunks[2].metadata["level"] == 3


def test_boundary_marker_overlap():
    """Test overlap for boundary marker chunking."""
    text = """[[[PAGE 1]]]
This is the content of page one. Important context at the end.

[[[PAGE 2]]]
This is the content of page two.

[[[PAGE 3]]]
This is the content of page three."""

    chunker = BoundaryMarkerChunker("PAGE", overlap_chars=50)
    chunks = chunker.chunk(text, {})

    assert len(chunks) == 3

    # Second and third chunks should have overlap
    assert chunks[1].metadata.get("has_prefix_overlap") is True
    assert chunks[2].metadata.get("has_prefix_overlap") is True

    # First chunk has no overlap
    assert chunks[0].metadata.get("has_prefix_overlap") is None

    # Verify anchors
    assert chunks[0].metadata["anchor"] == "1"
    assert chunks[1].metadata["anchor"] == "2"
    assert chunks[2].metadata["anchor"] == "3"


def test_boundary_marker_no_overlap():
    """Test boundary marker chunking without overlap."""
    text = """[[[SLIDE 1]]]
Slide one content.

[[[SLIDE 2]]]
Slide two content."""

    chunker = BoundaryMarkerChunker("SLIDE", overlap_chars=0)
    chunks = chunker.chunk(text, {})

    assert len(chunks) == 2
    assert all(c.metadata.get("has_prefix_overlap") is None for c in chunks)


def test_markdown_empty_sections():
    """Test handling of empty sections."""
    text = """# Section 1
Content here.

# Empty Section

# Section 3
More content."""

    chunker = MarkdownChunker(overlap_chars=50, max_chunk_size=2000)
    chunks = chunker.chunk(text, {})

    # Empty section will have overlap from previous, so it's included
    # This is acceptable behavior - the overlap provides context
    assert len(chunks) == 3
    assert chunks[0].metadata["heading"] == "Section 1"
    assert chunks[1].metadata["heading"] == "Empty Section"
    assert chunks[2].metadata["heading"] == "Section 3"

    # The empty section should have overlap from Section 1
    assert chunks[1].metadata.get("has_prefix_overlap") is True


def test_markdown_no_headings():
    """Test handling of text without headings."""
    text = "This is just plain text without any headings."

    chunker = MarkdownChunker(overlap_chars=50, max_chunk_size=2000)
    chunks = chunker.chunk(text, {})

    assert len(chunks) == 1
    assert chunks[0].anchor_ref == "ROOT"


def test_markdown_paragraph_boundary_split():
    """Test that large text splits at paragraph boundaries."""
    # Create text with paragraph breaks
    paragraphs = ["Paragraph {}. " * 50 for i in range(10)]
    text = "# Long Section\n\n" + "\n\n".join(paragraphs)

    chunker = MarkdownChunker(overlap_chars=100, max_chunk_size=1500)
    chunks = chunker.chunk(text, {})

    # Should be split into multiple chunks
    assert len(chunks) > 1

    # Check that splits happened (look for paragraph boundaries)
    for chunk in chunks:
        if chunk.metadata.get("is_split"):
            # Chunk should end reasonably (not mid-word)
            assert not chunk.text.endswith(" P")  # Shouldn't cut off "Paragraph"
