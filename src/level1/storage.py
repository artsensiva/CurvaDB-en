"""LMDB-based storage module for CurvaDB Level 1.

This module provides a key-value storage layer using LMDB (Lightning
Memory-Mapped Database) for efficient persistence and range queries.
"""

from typing import Optional, List, Tuple, Any
import struct
import lmdb
import msgpack


class LMDBStorage:
    """Key-value storage using LMDB with support for range queries.

    LMDB is a high-performance embedded database that uses memory-mapped files
    for speed and supports ordered keys for efficient range queries.

    Args:
        db_path: Path to the database directory.
        map_size: Maximum size of the database in bytes (default: 10GB).
        readonly: Open database in read-only mode.

    Example:
        >>> storage = LMDBStorage("./my_db")
        >>> # Store data
        >>> storage.put(1000, {"doc_id": "doc1", "text": "Hello"})
        >>> # Retrieve data
        >>> data = storage.get(1000)
        >>> data["doc_id"]
        'doc1'
        >>> # Range query
        >>> results = storage.range_query(990, 1010)
        >>> storage.close()
    """

    def __init__(
        self,
        db_path: str,
        map_size: int = 100 * 1024**2,  # 100MB default (was 10GB!)
        readonly: bool = False
    ) -> None:
        """Initialize LMDB storage.

        Args:
            db_path: Directory path for the database.
            map_size: Maximum database size in bytes.
            readonly: Whether to open in read-only mode.
        """
        self.db_path = db_path
        self.map_size = map_size
        self.readonly = readonly

        # Open LMDB environment
        self.env = lmdb.open(
            db_path,
            map_size=map_size,
            readonly=readonly,
            create=not readonly,
            max_dbs=0,
            metasync=False,  # Better performance
            sync=False,  # Better performance, manual sync with sync() method
            writemap=True  # Better performance on some systems
        )

    def put(self, key: int, value: Any) -> None:
        """Store a key-value pair.

        Args:
            key: Integer key (typically a Hilbert distance).
            value: Any Python object that can be serialized with msgpack.

        Raises:
            ValueError: If key is negative or too large.
        """
        if key < 0:
            raise ValueError(f"Key must be non-negative, got {key}")

        # Convert integer to bytes (big-endian for sorting, FIXED 16 bytes)
        # Fixed length ensures proper sorting in LMDB
        key_bytes = key.to_bytes(16, byteorder='big', signed=False)

        # Serialize value with msgpack
        value_bytes = msgpack.packb(value, use_bin_type=True)

        # Write to database
        with self.env.begin(write=True) as txn:
            txn.put(key_bytes, value_bytes)

    def get(self, key: int) -> Optional[Any]:
        """Retrieve value by key.

        Args:
            key: Integer key to look up.

        Returns:
            Deserialized value if key exists, None otherwise.
        """
        key_bytes = key.to_bytes(16, byteorder='big', signed=False)

        with self.env.begin() as txn:
            value_bytes = txn.get(key_bytes)

        if value_bytes is None:
            return None

        return msgpack.unpackb(value_bytes, raw=False)

    def range_query(
        self,
        start_key: int,
        end_key: int,
        limit: Optional[int] = None
    ) -> List[Tuple[int, Any]]:
        """Query all key-value pairs in a range.

        This is the core operation for Hilbert curve search - we query a range
        of Hilbert distances to find nearby points in N-dimensional space.

        Args:
            start_key: Start of range (inclusive).
            end_key: End of range (inclusive).
            limit: Maximum number of results to return (None = no limit).

        Returns:
            List of (key, value) tuples in sorted order by key.

        Example:
            >>> storage = LMDBStorage("./db")
            >>> storage.put(100, {"text": "doc1"})
            >>> storage.put(150, {"text": "doc2"})
            >>> storage.put(200, {"text": "doc3"})
            >>> results = storage.range_query(140, 210)
            >>> len(results)
            2
            >>> results[0][1]["text"]
            'doc2'
        """
        # Use fixed 16 bytes for proper sorting
        start_bytes = max(0, start_key).to_bytes(16, byteorder='big', signed=False)
        end_bytes = end_key.to_bytes(16, byteorder='big', signed=False)

        results = []

        with self.env.begin() as txn:
            cursor = txn.cursor()

            # Position cursor at start of range
            if not cursor.set_range(start_bytes):
                # No keys >= start_key
                return results

            # Iterate through range
            for key_bytes, value_bytes in cursor:
                if key_bytes > end_bytes:
                    break

                # Unpack key and value
                key = int.from_bytes(key_bytes, byteorder='big', signed=False)
                value = msgpack.unpackb(value_bytes, raw=False)

                results.append((key, value))

                if limit is not None and len(results) >= limit:
                    break

        return results

    def delete(self, key: int) -> bool:
        """Delete a key-value pair.

        Args:
            key: Key to delete.

        Returns:
            True if key was deleted, False if key didn't exist.
        """
        key_bytes = key.to_bytes(16, byteorder='big', signed=False)

        with self.env.begin(write=True) as txn:
            return txn.delete(key_bytes)

    def count(self) -> int:
        """Count total number of entries in the database.

        Returns:
            Number of key-value pairs.
        """
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def sync(self) -> None:
        """Manually sync the database to disk.

        Call this periodically to ensure data is persisted, especially
        if you disabled automatic syncing for performance.
        """
        self.env.sync()

    def close(self) -> None:
        """Close the database and release resources.

        Always call this when done using the database.
        """
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self) -> "LMDBStorage":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures database is closed."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the storage."""
        try:
            count = self.count()
        except:
            count = "?"
        return f"LMDBStorage(path='{self.db_path}', entries={count})"
