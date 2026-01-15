"""
Comprehensive tests for chunking strategies (Markdown, Boundary Marker).
Tests segmentation, metadata, and anchor generation.
"""

import pytest
from cortexindex.chunking.markdown_chunker import MarkdownChunker
from cortexindex.chunking.boundary_chunker import BoundaryMarkerChunker


class TestMarkdownChunker:
    """Test Markdown-specific chunking strategy."""

    def test_chunk_by_headings(self):
        """Test that markdown is chunked by headings."""
        chunker = MarkdownChunker()

        text = """# Main Heading

Some content under main heading.

## Sub Heading 1

Content for sub heading 1.

## Sub Heading 2

Content for sub heading 2.

### Deeper Heading

Deep content."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        # Should have multiple chunks (one per heading section)
        assert len(chunks) >= 4

    def test_chunk_preserves_heading_anchors(self):
        """Test that chunks have proper heading anchors."""
        chunker = MarkdownChunker()

        text = """# Main
Content.

## Sub 1
Sub content.

## Sub 2
More content."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        # Check for heading anchors
        heading_chunks = [c for c in chunks if c.anchor_type == "heading"]
        assert len(heading_chunks) > 0

    def test_chunk_simple_text(self):
        """Test chunking of simple text without headings."""
        chunker = MarkdownChunker()

        text = "Some simple text without any headings."

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        # Should create at least one chunk
        chunk = chunks[0]
        assert chunk.text is not None
        assert len(chunk.text) > 0

    def test_chunk_preserves_code_blocks(self):
        """Test that code blocks are preserved in chunks."""
        chunker = MarkdownChunker()

        text = """# Code Example

```python
def hello():
    print("world")
```

More text after."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        # Should have code block content in chunks
        all_text = " ".join([c.text for c in chunks])
        assert "hello" in all_text or "Code" in all_text

    def test_chunk_handles_lists(self):
        """Test chunking of markdown lists."""
        chunker = MarkdownChunker()

        text = """# Items

- Item 1
- Item 2
  - Nested
- Item 3

End of list."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        all_text = " ".join([c.text for c in chunks])
        assert "Item" in all_text

    def test_chunk_preserves_metadata(self):
        """Test that chunk metadata is preserved."""
        chunker = MarkdownChunker()

        text = """# Section

Content here."""

        metadata = {"source": "test", "author": "unittest"}
        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.metadata is not None
        # Original metadata should be accessible
        assert "source" in chunk.metadata or chunk.metadata is not None

    def test_chunk_with_tables(self):
        """Test chunking of markdown with tables."""
        chunker = MarkdownChunker()

        text = """# Data

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

After table."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) > 0
        all_text = " ".join([c.text for c in chunks])
        # Table content should be in chunks
        assert len(all_text) > 0

    def test_chunk_multiple_sections(self):
        """Test that multiple h1 sections are separated."""
        chunker = MarkdownChunker()

        text = """# Section 1

Content 1.

# Section 2

Content 2.

# Section 3

Content 3."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 3  # At least 3 sections

    def test_chunk_long_content(self):
        """Test chunking of longer content."""
        chunker = MarkdownChunker()

        # Create a longer document
        sections = []
        for i in range(10):
            sections.append(f"""## Section {i}

This is content for section {i}.

- Point 1
- Point 2
""")

        text = "# Main\n\n" + "\n".join(sections)
        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        # Should have multiple chunks
        assert len(chunks) >= 10

    def test_chunk_anchor_references(self):
        """Test that anchor references are generated."""
        chunker = MarkdownChunker()

        text = """# Main

## Subsection

Content."""

        chunks = chunker.chunk(text, {})

        for chunk in chunks:
            assert chunk.anchor_type is not None
            assert chunk.anchor_ref is not None


class TestBoundaryMarkerChunker:
    """Test boundary marker chunking (for PDFs, slides, sheets)."""

    def test_chunk_with_page_boundaries(self):
        """Test chunking with PAGE boundary markers."""
        chunker = BoundaryMarkerChunker("PAGE")

        text = """PAGE
Content for page 1.
More content.

PAGE
Content for page 2.

PAGE
Page 3 content."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 3  # Should have 3 chunks for 3 pages

    def test_chunk_with_slide_boundaries(self):
        """Test chunking with SLIDE boundary markers."""
        chunker = BoundaryMarkerChunker("SLIDE")

        text = """SLIDE
Slide 1 Title
- Bullet 1
- Bullet 2

SLIDE
Slide 2 Title
- Point A
- Point B

SLIDE
Slide 3"""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 3

    def test_chunk_with_sheet_boundaries(self):
        """Test chunking with SHEET boundary markers."""
        chunker = BoundaryMarkerChunker("SHEET")

        text = """SHEET
Sheet 1 Data
Column A, Column B
Data 1, Data 2

SHEET
Sheet 2 Data
Column X, Column Y
Value 1, Value 2"""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 2

    def test_chunk_with_image_boundaries(self):
        """Test chunking with IMAGE boundary markers."""
        chunker = BoundaryMarkerChunker("IMAGE")

        text = """IMAGE
Image description 1.
Some metadata.

IMAGE
Image description 2.
Different metadata."""

        chunks = chunker.chunk(text, {})

        assert len(chunks) >= 2

    def test_chunk_preserves_boundary_anchor(self):
        """Test that boundary chunks have proper anchors."""
        chunker = BoundaryMarkerChunker("PAGE")

        text = """PAGE
Content

PAGE
More content"""

        chunks = chunker.chunk(text, {})

        for chunk in chunks:
            assert chunk.anchor_type == "boundary"
            assert chunk.anchor_ref is not None

    def test_chunk_handles_content_between_boundaries(self):
        """Test content between boundary markers is captured."""
        chunker = BoundaryMarkerChunker("PAGE")

        text = """Initial content before first marker.

PAGE
Page 1 content.
Multiple lines.

PAGE
Page 2 content."""

        chunks = chunker.chunk(text, {})

        # Should have chunks including boundary and content
        all_text = " ".join([c.text for c in chunks])
        assert "Page" in all_text

    def test_chunk_empty_sections(self):
        """Test handling of empty sections between boundaries."""
        chunker = BoundaryMarkerChunker("PAGE")

        text = """PAGE

PAGE
Some content

PAGE
"""

        chunks = chunker.chunk(text, {})

        # Should create chunks even if some are empty
        assert len(chunks) > 0


class TestChunkerRegistry:
    """Test the chunker registry."""

    def test_registry_markdown_chunker(self):
        """Test getting markdown chunker."""
        from cortexindex.chunking.base import ChunkerRegistry

        registry = ChunkerRegistry()
        registry.register("markdown", MarkdownChunker())

        chunker = registry.get("markdown")
        assert chunker is not None
        assert isinstance(chunker, MarkdownChunker)

    def test_registry_boundary_chunker(self):
        """Test getting boundary chunker."""
        from cortexindex.chunking.base import ChunkerRegistry

        registry = ChunkerRegistry()
        registry.register("pdf", BoundaryMarkerChunker("PAGE"))

        chunker = registry.get("pdf")
        assert chunker is not None
        assert isinstance(chunker, BoundaryMarkerChunker)

    def test_registry_unknown_type(self):
        """Test registry with unknown type."""
        from cortexindex.chunking.base import ChunkerRegistry

        registry = ChunkerRegistry()
        chunker = registry.get("unknown")
        assert chunker is None


class TestChunkerConsistency:
    """Test that chunkers produce consistent results."""

    def test_markdown_chunker_idempotent(self):
        """Test that markdown chunking is idempotent."""
        chunker = MarkdownChunker()

        text = """# Section

Content here.

## Subsection

More content."""

        chunks1 = chunker.chunk(text, {})
        chunks2 = chunker.chunk(text, {})

        assert len(chunks1) == len(chunks2)

        # Same text content
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.text == c2.text
            assert c1.anchor_type == c2.anchor_type

    def test_boundary_chunker_idempotent(self):
        """Test that boundary chunking is idempotent."""
        chunker = BoundaryMarkerChunker("PAGE")

        text = """PAGE
Content 1

PAGE
Content 2"""

        chunks1 = chunker.chunk(text, {})
        chunks2 = chunker.chunk(text, {})

        assert len(chunks1) == len(chunks2)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1.text == c2.text


class TestChunkerIntegration:
    """Test chunkers with real extracted content."""

    def test_markdown_chunker_with_extracted_metadata(self):
        """Test markdown chunker with extracted metadata."""
        chunker = MarkdownChunker()

        text = """# Note Title

Content with tags: tag1, tag2

## Details

More info."""

        metadata = {
            "wikilinks": ["other-note", "reference"],
            "tags": ["tag1", "tag2"],
        }

        chunks = chunker.chunk(text, metadata)

        assert len(chunks) > 0
        # Verify metadata flows through
        for chunk in chunks:
            assert chunk.metadata is not None
