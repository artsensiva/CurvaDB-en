"""Efficient embedding storage using memory-mapped numpy arrays.

This module provides compact storage for embeddings separate from the main
database, using numpy memory-mapped files for fast access and low memory usage.
"""

import numpy as np
import os
from typing import List, Optional


class EmbeddingStore:
    """Memory-mapped storage for embeddings.

    Stores embeddings in a compact binary format using numpy mmap, allowing
    fast random access without loading everything into memory.

    Args:
        store_path: Directory to store embedding files.
        embedding_dim: Dimensionality of embeddings.

    Example:
        >>> store = EmbeddingStore('./embeddings', embedding_dim=384)
        >>> # Add embeddings
        >>> embeddings = np.random.randn(100, 384)
        >>> store.add_batch(embeddings)
        >>> # Retrieve embeddings
        >>> emb = store.get(0)  # Get first embedding
        >>> emb.shape
        (384,)
        >>> store.close()
    """

    def __init__(self, store_path: str, embedding_dim: int) -> None:
        """Initialize embedding store."""
        self.store_path = store_path
        self.embedding_dim = embedding_dim

        os.makedirs(store_path, exist_ok=True)

        # Files
        self.embeddings_file = os.path.join(store_path, 'embeddings.npy')
        self.metadata_file = os.path.join(store_path, 'metadata.npz')

        # Load or create
        if os.path.exists(self.embeddings_file):
            # Load existing
            metadata = np.load(self.metadata_file)
            self.count = int(metadata['count'])

            # Calculate capacity from file size
            file_size = os.path.getsize(self.embeddings_file)
            capacity = file_size // (embedding_dim * 4)  # 4 bytes per float32

            self.embeddings = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='r+',
                shape=(capacity, embedding_dim)
            )
        else:
            # Create new with minimal initial capacity (will expand as needed)
            initial_capacity = 1000  # Start small
            self.embeddings = np.memmap(
                self.embeddings_file,
                dtype=np.float32,
                mode='w+',
                shape=(initial_capacity, embedding_dim)
            )
            self.count = 0
            self._save_metadata()

    def add_batch(self, embeddings: np.ndarray) -> List[int]:
        """Add a batch of embeddings.

        Args:
            embeddings: Array of shape (n_embeddings, embedding_dim).

        Returns:
            List of indices where embeddings were stored.
        """
        n_new = embeddings.shape[0]

        # Ensure capacity
        if self.count + n_new > len(self.embeddings):
            self._expand_capacity(self.count + n_new)

        # Store embeddings
        start_idx = self.count
        end_idx = self.count + n_new
        self.embeddings[start_idx:end_idx] = embeddings.astype(np.float32)

        # Update count
        indices = list(range(start_idx, end_idx))
        self.count = end_idx
        self._save_metadata()

        return indices

    def get(self, index: int) -> np.ndarray:
        """Get embedding by index.

        Args:
            index: Index of embedding to retrieve.

        Returns:
            Embedding vector of shape (embedding_dim,).
        """
        if index >= self.count:
            raise IndexError(f"Index {index} out of range (count={self.count})")
        return self.embeddings[index].copy()

    def get_batch(self, indices: List[int]) -> np.ndarray:
        """Get multiple embeddings by indices.

        Args:
            indices: List of indices to retrieve.

        Returns:
            Array of shape (len(indices), embedding_dim).
        """
        if not indices:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        max_idx = max(indices)
        if max_idx >= self.count:
            raise IndexError(f"Index {max_idx} out of range (count={self.count})")

        return self.embeddings[indices].copy()

    def get_all(self) -> np.ndarray:
        """Get all stored embeddings.

        Returns:
            Array of shape (count, embedding_dim).
        """
        return self.embeddings[:self.count].copy()

    def _expand_capacity(self, new_capacity: int) -> None:
        """Expand storage capacity."""
        # Round up to nearest 1000
        new_capacity = ((new_capacity // 1000) + 1) * 1000

        # Create new larger array
        new_file = self.embeddings_file + '.tmp'
        new_embeddings = np.memmap(
            new_file,
            dtype=np.float32,
            mode='w+',
            shape=(new_capacity, self.embedding_dim)
        )

        # Copy existing data
        new_embeddings[:self.count] = self.embeddings[:self.count]

        # Replace old with new
        del self.embeddings
        os.replace(new_file, self.embeddings_file)

        # Reopen as memmap
        self.embeddings = np.memmap(
            self.embeddings_file,
            dtype=np.float32,
            mode='r+',
            shape=(new_capacity, self.embedding_dim)
        )

    def _save_metadata(self) -> None:
        """Save metadata (count, etc.)."""
        np.savez(
            self.metadata_file,
            count=self.count,
            embedding_dim=self.embedding_dim
        )

    def size_bytes(self) -> int:
        """Get total storage size in bytes."""
        total = 0
        if os.path.exists(self.embeddings_file):
            total += os.path.getsize(self.embeddings_file)
        if os.path.exists(self.metadata_file):
            total += os.path.getsize(self.metadata_file)
        return total

    def close(self) -> None:
        """Close and flush the store."""
        if hasattr(self, 'embeddings'):
            self.embeddings.flush()
            del self.embeddings

    def __len__(self) -> int:
        """Number of stored embeddings."""
        return self.count

    def __repr__(self) -> str:
        """String representation."""
        size_mb = self.size_bytes() / 1024 / 1024
        return f"EmbeddingStore(count={self.count}, dim={self.embedding_dim}, size={size_mb:.1f}MB)"
