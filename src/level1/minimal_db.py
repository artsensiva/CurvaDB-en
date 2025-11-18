"""Main database class for CurvaDB Level 1.

This module provides the primary API for the Minimal Hilbert Vector Database.
"""

from typing import List, Tuple, Optional
import numpy as np
import os

from .embedder import TextEmbedder
from .reducer import DimensionalityReducer
from .hilbert_index import HilbertIndexer
from .storage import LMDBStorage
from .search import SearchEngine
from .embedding_store import EmbeddingStore


class MinimalCurveDB:
    """Minimal Hilbert Curve-based Vector Database.

    This database uses space-filling Hilbert curves to index high-dimensional
    text embeddings for efficient semantic search.

    Pipeline:
    1. Text → Embeddings (SentenceTransformer)
    2. Embeddings → Reduced dims (PCA)
    3. Reduced → Normalized integers
    4. Integers → Hilbert distance (1D index)
    5. Store in LMDB key-value store

    Search:
    1. Query → Hilbert distance
    2. Range query around that distance
    3. Rerank by cosine similarity
    4. Return top-K results

    Args:
        db_path: Path to database directory.
        p: Hilbert curve precision (bits per dimension). Default: 16.
        n: Number of reduced dimensions. Default: 8.
        model_name: SentenceTransformer model name. Default: 'all-MiniLM-L6-v2'.

    Example:
        >>> db = MinimalCurveDB(db_path='./my_db')
        >>> # Add documents
        >>> db.add_batch(
        ...     doc_ids=['doc1', 'doc2'],
        ...     texts=['Machine learning', 'Deep learning']
        ... )
        >>> # Search
        >>> results = db.search('neural networks', k=2)
        >>> for doc_id, text, score in results:
        ...     print(f"{doc_id}: {text} ({score:.3f})")
        >>> db.close()
    """

    def __init__(
        self,
        db_path: str = "./curve_db",
        p: int = 10,
        n: int = 12,  # Balanced for recall and performance
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        """Initialize the database."""
        self.db_path = db_path
        self.p = p
        self.n = n
        self.model_name = model_name

        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize components
        self.embedder = TextEmbedder(model_name=model_name)
        self.reducer = DimensionalityReducer(
            n_components=n,
            max_val=(2 ** p) - 1
        )
        self.hilbert = HilbertIndexer(p=p, n=n)
        self.storage = LMDBStorage(db_path=db_path)
        self.embedding_store = EmbeddingStore(
            store_path=os.path.join(db_path, "embeddings"),
            embedding_dim=self.embedder.get_dimension()
        )
        self.search_engine = SearchEngine()

        # Track if reducer is fitted
        self.fitted = False

        # Try to load existing reducer model
        reducer_path = os.path.join(db_path, "reducer.pkl")
        if os.path.exists(reducer_path):
            try:
                self.reducer = DimensionalityReducer.load(reducer_path)
                self.fitted = True
            except Exception as e:
                print(f"Warning: Could not load reducer model: {e}")

    def add_batch(
        self,
        doc_ids: List[str],
        texts: List[str],
        show_progress: bool = False
    ) -> None:
        """Add multiple documents to the database.

        Args:
            doc_ids: List of unique document identifiers.
            texts: List of text documents to add.
            show_progress: Whether to show progress bar during embedding.

        Raises:
            ValueError: If doc_ids and texts have different lengths.
            RuntimeError: If database is in read-only mode.

        Example:
            >>> db = MinimalCurveDB()
            >>> db.add_batch(
            ...     doc_ids=['d1', 'd2', 'd3'],
            ...     texts=['First doc', 'Second doc', 'Third doc']
            ... )
        """
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and texts ({len(texts)}) "
                f"must have same length"
            )

        if len(texts) == 0:
            return

        # 1. Embed all texts
        embeddings = self.embedder.encode(texts, show_progress=show_progress)

        # 2. Fit reducer on first batch if not fitted yet
        if not self.fitted:
            if len(embeddings) < self.n:
                raise ValueError(
                    f"First batch must have at least {self.n} documents "
                    f"to fit PCA, got {len(embeddings)}"
                )
            self.reducer.fit(embeddings)
            self.fitted = True

            # Save reducer model
            reducer_path = os.path.join(self.db_path, "reducer.pkl")
            self.reducer.save(reducer_path)

        # 3. Reduce dimensions and normalize
        reduced = self.reducer.transform(embeddings)

        # 4. Store embeddings separately (compact storage)
        emb_indices = self.embedding_store.add_batch(embeddings)

        # 5. Convert to Hilbert distances
        hilbert_distances = self.hilbert.points_to_distances(reduced.tolist())

        # 6. Store in LMDB (without embeddings - much smaller!)
        for doc_id, text, hilbert_dist, emb_idx in zip(
            doc_ids, texts, hilbert_distances, emb_indices
        ):
            # Store only metadata, not embeddings
            doc_data = {
                "doc_id": doc_id,
                "text": text,
                "emb_idx": emb_idx  # Index into embedding store
            }
            self.storage.put(hilbert_dist, doc_data)

        # Sync to disk
        self.storage.sync()

    def search(
        self,
        query: str,
        k: int = 10,
        radius: Optional[int] = None,
        rerank: bool = True
    ) -> List[Tuple[str, str, float]]:
        """Search for documents similar to the query.

        Args:
            query: Text query to search for.
            k: Number of top results to return.
            radius: Search radius around query's Hilbert distance.
                   If None, auto-computed as max(1000, k * 100).
            rerank: Whether to rerank by cosine similarity (recommended).

        Returns:
            List of (doc_id, text, similarity_score) tuples,
            sorted by relevance (highest score first).

        Example:
            >>> db = MinimalCurveDB()
            >>> db.add_batch(['d1'], ['Machine learning is great'])
            >>> results = db.search('artificial intelligence', k=1)
            >>> doc_id, text, score = results[0]
            >>> isinstance(doc_id, str) and isinstance(text, str)
            True
        """
        if not self.fitted:
            raise RuntimeError(
                "Database is empty. Add documents with add_batch() first."
            )

        # Default radius: 20% of max distance for better recall
        # This ensures we search a reasonable portion of the space
        if radius is None:
            adaptive_radius = self.hilbert.get_max_distance() // 5  # 20% of space
            radius = max(adaptive_radius, k * 10000)

        # 1. Embed query
        query_embedding = self.embedder.encode([query])[0]

        # 2. Reduce and normalize query
        query_reduced = self.reducer.transform(query_embedding.reshape(1, -1))[0]

        # 3. Get Hilbert distance
        query_distance = self.hilbert.points_to_distances([query_reduced.tolist()])[0]

        # 4. Compute search range
        start_dist, end_dist = self.search_engine.compute_range(
            query_distance,
            radius,
            self.hilbert.get_max_distance()
        )

        # 5. Range query to get candidates
        # Get more candidates than k for better recall after reranking
        candidate_limit = k * 3 if rerank else k
        candidates = self.storage.range_query(
            start_dist,
            end_dist,
            limit=candidate_limit
        )

        # Auto-expand radius if too few candidates found
        if len(candidates) < k:
            # Try with expanded radius (50% of space)
            expanded_radius = self.hilbert.get_max_distance() // 2
            start_dist, end_dist = self.search_engine.compute_range(
                query_distance,
                expanded_radius,
                self.hilbert.get_max_distance()
            )
            candidates = self.storage.range_query(
                start_dist,
                end_dist,
                limit=candidate_limit
            )

        if not candidates:
            return []  # No results found

        # 6. Rerank by cosine similarity
        if rerank and candidates:
            # Retrieve embeddings from store (efficient batch operation)
            emb_indices = [doc_data["emb_idx"] for _, doc_data in candidates]
            candidate_embeddings = self.embedding_store.get_batch(emb_indices)

            # Add embeddings to candidate data for reranking
            for i, (dist, doc_data) in enumerate(candidates):
                doc_data["embedding"] = candidate_embeddings[i]

            scored_results = self.search_engine.rerank_by_similarity(
                query_embedding,
                candidates,
                k=k,
                embedding_key="embedding"
            )

            # Format results
            results = [
                (doc_data["doc_id"], doc_data["text"], score)
                for _, doc_data, score in scored_results
            ]
        else:
            # Return top k candidates without reranking
            results = [
                (doc_data["doc_id"], doc_data["text"], 0.0)
                for _, doc_data in candidates[:k]
            ]

        return results

    def count(self) -> int:
        """Get the total number of documents in the database.

        Returns:
            Number of documents stored.
        """
        return self.storage.count()

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        return {
            "db_path": self.db_path,
            "model": self.model_name,
            "dimensions": self.n,
            "precision": self.p,
            "embedding_dim": self.embedder.get_dimension(),
            "total_docs": self.count(),
            "fitted": self.fitted
        }

    def close(self) -> None:
        """Close the database and release resources.

        Always call this when done using the database.
        """
        if self.storage is not None:
            self.storage.close()
        if self.embedding_store is not None:
            self.embedding_store.close()

    def __enter__(self) -> "MinimalCurveDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"MinimalCurveDB(docs={stats['total_docs']}, "
            f"model='{stats['model']}', "
            f"dims={stats['dimensions']})"
        )
