"""Tests for query instruction prefix functionality."""
import pytest
from unittest.mock import Mock, patch
import numpy as np


def test_sentence_transformers_query_prefix():
    """Test that SentenceTransformersEmbedder applies prefix to queries."""
    from ragtriever.embeddings.sentence_transformers import SentenceTransformersEmbedder

    # Mock the SentenceTransformer to avoid loading actual model
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        embedder = SentenceTransformersEmbedder(
            model_id="test-model",
            device="cpu",
            use_query_prefix=True,
            query_prefix="Search query: "
        )

        # Test query embedding
        embedder.embed_query("test query")

        # Verify prefix was added
        calls = mock_model.encode.call_args_list
        # First call is for dimension probe
        # Second call should have prefix
        assert len(calls) >= 2
        query_call = calls[-1]
        assert query_call[0][0][0] == "Search query: test query"


def test_sentence_transformers_document_no_prefix():
    """Test that document embeddings don't get prefix."""
    from ragtriever.embeddings.sentence_transformers import SentenceTransformersEmbedder

    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st.return_value = mock_model

        embedder = SentenceTransformersEmbedder(
            model_id="test-model",
            device="cpu",
            use_query_prefix=True,
            query_prefix="Search query: "
        )

        # Test document embedding
        embedder.embed_texts(["doc1", "doc2"])

        # Get the last call (document embedding)
        calls = mock_model.encode.call_args_list
        doc_call = calls[-1]

        # Documents should not have prefix
        assert doc_call[0][0] == ["doc1", "doc2"]


def test_sentence_transformers_prefix_disabled():
    """Test that prefix can be disabled."""
    from ragtriever.embeddings.sentence_transformers import SentenceTransformersEmbedder

    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        embedder = SentenceTransformersEmbedder(
            model_id="test-model",
            device="cpu",
            use_query_prefix=False,  # Disabled
            query_prefix="Search query: "
        )

        # Test query embedding
        embedder.embed_query("test query")

        # Verify no prefix was added
        calls = mock_model.encode.call_args_list
        query_call = calls[-1]
        assert query_call[0][0][0] == "test query"


def test_ollama_query_prefix():
    """Test that OllamaEmbedder applies prefix to queries."""
    from ragtriever.embeddings.ollama import OllamaEmbedder

    embedder = OllamaEmbedder(
        model_id="test-model",
        use_query_prefix=True,
        query_prefix="Search: "
    )

    # Mock the _call method
    with patch.object(embedder, '_call') as mock_call:
        mock_call.return_value = [0.1, 0.2, 0.3]

        embedder.embed_query("test query")

        # Verify prefix was added
        mock_call.assert_called_once_with("Search: test query")


def test_ollama_document_no_prefix():
    """Test that Ollama document embeddings don't get prefix."""
    from ragtriever.embeddings.ollama import OllamaEmbedder

    embedder = OllamaEmbedder(
        model_id="test-model",
        use_query_prefix=True,
        query_prefix="Search: "
    )

    # Mock the _call method
    with patch.object(embedder, '_call') as mock_call:
        mock_call.return_value = [0.1, 0.2, 0.3]

        embedder.embed_texts(["doc1", "doc2"])

        # Verify no prefix on documents
        calls = mock_call.call_args_list
        assert calls[0][0][0] == "doc1"
        assert calls[1][0][0] == "doc2"


def test_ollama_prefix_disabled():
    """Test that Ollama prefix can be disabled."""
    from ragtriever.embeddings.ollama import OllamaEmbedder

    embedder = OllamaEmbedder(
        model_id="test-model",
        use_query_prefix=False,
        query_prefix="Search: "
    )

    with patch.object(embedder, '_call') as mock_call:
        mock_call.return_value = [0.1, 0.2, 0.3]

        embedder.embed_query("test query")

        # Verify no prefix when disabled
        mock_call.assert_called_once_with("test query")


def test_bge_default_prefix():
    """Test that the default BGE prefix is correct."""
    from ragtriever.embeddings.sentence_transformers import SentenceTransformersEmbedder

    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        # Use default prefix
        embedder = SentenceTransformersEmbedder(
            model_id="test-model",
            device="cpu",
        )

        embedder.embed_query("test")

        # Check that default BGE prefix is used
        calls = mock_model.encode.call_args_list
        query_call = calls[-1]
        assert query_call[0][0][0].startswith("Represent this sentence for searching relevant passages:")
