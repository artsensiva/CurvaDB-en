"""Tests for TextEmbedder module."""

import pytest
import numpy as np

pytest.importorskip("sentence_transformers", reason="legacy level1 dep not installed in venv")

from level1.embedder import TextEmbedder


def test_embedder_initialization():
    """Test that embedder initializes correctly."""
    embedder = TextEmbedder()
    assert embedder.embedding_dim > 0
    assert embedder.model_name == "all-MiniLM-L6-v2"


def test_encode_single_text():
    """Test encoding a single text."""
    embedder = TextEmbedder()
    text = "Hello world"
    embedding = embedder.encode(text)

    assert embedding.shape == (1, embedder.embedding_dim)
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64


def test_encode_multiple_texts():
    """Test encoding multiple texts."""
    embedder = TextEmbedder()
    texts = ["Hello world", "Machine learning", "Deep learning"]
    embeddings = embedder.encode(texts)

    assert embeddings.shape == (3, embedder.embedding_dim)
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64


def test_encode_empty_list():
    """Test encoding empty list."""
    embedder = TextEmbedder()
    embeddings = embedder.encode([])

    assert embeddings.shape[0] == 0


def test_get_dimension():
    """Test get_dimension method."""
    embedder = TextEmbedder()
    dim = embedder.get_dimension()

    assert dim == 384  # all-MiniLM-L6-v2 has 384 dimensions
    assert dim == embedder.embedding_dim


def test_embedder_repr():
    """Test string representation."""
    embedder = TextEmbedder()
    repr_str = repr(embedder)

    assert "TextEmbedder" in repr_str
    assert "all-MiniLM-L6-v2" in repr_str
