"""Search engine module for CurvaDB Level 1.

This module implements two-stage search:
1. Approximate search via Hilbert range query (fast)
2. Reranking by cosine similarity (accurate)
"""

from typing import List, Tuple, Any
import numpy as np


class SearchEngine:
    """Two-stage search engine combining Hilbert indexing and cosine similarity.

    The search process:
    1. Convert query to Hilbert distance
    2. Perform range query around that distance (fast approximate search)
    3. Rerank candidates by actual cosine similarity (precision)
    4. Return top-K results

    This approach balances speed and accuracy.

    Example:
        >>> engine = SearchEngine()
        >>> query_emb = np.array([0.1, 0.2, 0.3])
        >>> candidates = [
        ...     (100, {"text": "doc1", "embedding": np.array([0.11, 0.21, 0.31])}),
        ...     (150, {"text": "doc2", "embedding": np.array([0.9, 0.8, 0.7])})
        ... ]
        >>> results = engine.rerank_by_similarity(query_emb, candidates, k=1)
        >>> results[0][1]["text"]
        'doc1'
    """

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity in range [-1, 1].
            Higher values mean more similar.

        Example:
            >>> v1 = np.array([1.0, 0.0, 0.0])
            >>> v2 = np.array([1.0, 0.0, 0.0])
            >>> SearchEngine.cosine_similarity(v1, v2)
            1.0
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        # Compute cosine similarity
        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and multiple vectors efficiently.

        Args:
            query: Query vector of shape (dim,).
            vectors: Matrix of vectors of shape (n_vectors, dim).

        Returns:
            Array of similarities of shape (n_vectors,).

        Example:
            >>> query = np.array([1.0, 0.0])
            >>> vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            >>> sims = SearchEngine.batch_cosine_similarity(query, vecs)
            >>> sims.shape
            (3,)
        """
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Normalize all vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        vectors_norm = vectors / norms

        # Compute all similarities at once
        similarities = np.dot(vectors_norm, query_norm)

        return similarities

    @staticmethod
    def rerank_by_similarity(
        query_embedding: np.ndarray,
        candidates: List[Tuple[int, Any]],
        k: int,
        embedding_key: str = "embedding"
    ) -> List[Tuple[int, Any, float]]:
        """Rerank candidates by cosine similarity to query.

        Args:
            query_embedding: Query embedding vector.
            candidates: List of (hilbert_distance, document_data) tuples.
            k: Number of top results to return.
            embedding_key: Key in document_data dict where embedding is stored.

        Returns:
            List of (hilbert_distance, document_data, similarity_score) tuples,
            sorted by similarity (highest first), limited to top k.

        Example:
            >>> query = np.array([1.0, 0.0, 0.0])
            >>> candidates = [
            ...     (100, {"id": "doc1", "embedding": np.array([1.0, 0.1, 0.0])}),
            ...     (200, {"id": "doc2", "embedding": np.array([0.0, 1.0, 0.0])})
            ... ]
            >>> results = SearchEngine.rerank_by_similarity(query, candidates, k=2)
            >>> results[0][2] > results[1][2]  # First result more similar
            True
        """
        if not candidates:
            return []

        # Extract embeddings
        embeddings = []
        for _, doc_data in candidates:
            if embedding_key in doc_data:
                embeddings.append(doc_data[embedding_key])
            else:
                # If no embedding stored, use zero similarity
                embeddings.append(np.zeros_like(query_embedding))

        embeddings_array = np.array(embeddings)

        # Compute all similarities at once
        similarities = SearchEngine.batch_cosine_similarity(
            query_embedding,
            embeddings_array
        )

        # Create results with scores
        scored_results = [
            (hilbert_dist, doc_data, float(sim))
            for (hilbert_dist, doc_data), sim in zip(candidates, similarities)
        ]

        # Sort by similarity (descending)
        scored_results.sort(key=lambda x: x[2], reverse=True)

        # Return top k
        return scored_results[:k]

    @staticmethod
    def compute_range(
        query_distance: int,
        radius: int,
        max_distance: int
    ) -> Tuple[int, int]:
        """Compute search range around a query distance.

        Args:
            query_distance: Hilbert distance of query point.
            radius: Search radius around query.
            max_distance: Maximum possible Hilbert distance.

        Returns:
            Tuple of (start_distance, end_distance) for range query.

        Example:
            >>> start, end = SearchEngine.compute_range(1000, 100, 10000)
            >>> start
            900
            >>> end
            1100
        """
        start_distance = max(0, query_distance - radius)
        end_distance = min(max_distance, query_distance + radius)
        return start_distance, end_distance

    def __repr__(self) -> str:
        """String representation."""
        return "SearchEngine(two-stage: hilbert_range + cosine_rerank)"
