"""Tests for LMDBStorage module."""

import pytest
import tempfile
import shutil
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from level1.storage import LMDBStorage


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    path = tempfile.mkdtemp()
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


def test_storage_initialization(temp_db_path):
    """Test storage initialization."""
    storage = LMDBStorage(temp_db_path)
    assert storage.db_path == temp_db_path
    assert storage.count() == 0
    storage.close()


def test_put_and_get(temp_db_path):
    """Test putting and getting data."""
    storage = LMDBStorage(temp_db_path)

    # Put data
    storage.put(100, {"doc_id": "doc1", "text": "Hello"})

    # Get data
    data = storage.get(100)
    assert data is not None
    assert data["doc_id"] == "doc1"
    assert data["text"] == "Hello"

    storage.close()


def test_get_nonexistent(temp_db_path):
    """Test getting non-existent key."""
    storage = LMDBStorage(temp_db_path)
    data = storage.get(999)
    assert data is None
    storage.close()


def test_range_query(temp_db_path):
    """Test range query."""
    storage = LMDBStorage(temp_db_path)

    # Insert data
    storage.put(100, {"text": "doc1"})
    storage.put(200, {"text": "doc2"})
    storage.put(300, {"text": "doc3"})
    storage.put(400, {"text": "doc4"})

    # Range query
    results = storage.range_query(150, 350)

    assert len(results) == 2
    assert results[0][0] == 200  # First key in range
    assert results[1][0] == 300  # Second key in range

    storage.close()


def test_range_query_with_limit(temp_db_path):
    """Test range query with limit."""
    storage = LMDBStorage(temp_db_path)

    # Insert data
    for i in range(10):
        storage.put(i * 100, {"text": f"doc{i}"})

    # Range query with limit
    results = storage.range_query(0, 1000, limit=3)

    assert len(results) == 3

    storage.close()


def test_delete(temp_db_path):
    """Test deleting keys."""
    storage = LMDBStorage(temp_db_path)

    storage.put(100, {"text": "doc1"})
    assert storage.get(100) is not None

    # Delete
    deleted = storage.delete(100)
    assert deleted is True
    assert storage.get(100) is None

    # Delete non-existent
    deleted = storage.delete(999)
    assert deleted is False

    storage.close()


def test_count(temp_db_path):
    """Test counting entries."""
    storage = LMDBStorage(temp_db_path)

    assert storage.count() == 0

    storage.put(1, {"text": "doc1"})
    storage.put(2, {"text": "doc2"})

    assert storage.count() == 2

    storage.close()


def test_context_manager(temp_db_path):
    """Test using storage as context manager."""
    with LMDBStorage(temp_db_path) as storage:
        storage.put(100, {"text": "doc1"})
        assert storage.count() == 1

    # Should be closed after context
    # Opening again to verify data persisted
    with LMDBStorage(temp_db_path) as storage:
        assert storage.count() == 1


def test_large_hilbert_distances(temp_db_path):
    """Test storing very large Hilbert distances (>2^64)."""
    storage = LMDBStorage(temp_db_path)

    # Large distance that would overflow 64-bit
    large_dist = 2**70

    storage.put(large_dist, {"text": "doc_large"})
    data = storage.get(large_dist)

    assert data is not None
    assert data["text"] == "doc_large"

    storage.close()
