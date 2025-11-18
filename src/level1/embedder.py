"""Text embedding module for CurvaDB Level 1.

This module provides a simple wrapper around SentenceTransformer for
converting text documents into dense vector embeddings.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Converts text to dense vector embeddings using SentenceTransformer.

    Args:
        model_name: Name of the SentenceTransformer model to use.
            Default is 'all-MiniLM-L6-v2' (384 dimensions).
        device: Device to run the model on ('cpu', 'cuda', 'mps').
            If None, automatically selects the best available device.

    Example:
        >>> embedder = TextEmbedder()
        >>> embeddings = embedder.encode(["Hello world", "Machine learning"])
        >>> embeddings.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Union[str, None] = None
    ) -> None:
        """Initialize the text embedder with specified model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode text(s) into dense vector embeddings.

        Args:
            texts: Single text string or list of text strings to encode.
            batch_size: Number of texts to encode in parallel.
            show_progress: Whether to show a progress bar during encoding.

        Returns:
            numpy array of shape (n_texts, embedding_dim) with embeddings.
            If single text is provided, shape is (1, embedding_dim).

        Example:
            >>> embedder = TextEmbedder()
            >>> # Single text
            >>> emb = embedder.encode("Hello world")
            >>> emb.shape
            (1, 384)
            >>> # Multiple texts
            >>> embs = embedder.encode(["Text 1", "Text 2", "Text 3"])
            >>> embs.shape
            (3, 384)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Encode using SentenceTransformer
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        return embeddings

    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.

        Returns:
            Integer dimension of embedding vectors.
        """
        return self.embedding_dim

    def __repr__(self) -> str:
        """String representation of the embedder."""
        return f"TextEmbedder(model='{self.model_name}', dim={self.embedding_dim})"
