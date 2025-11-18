"""Dimensionality reduction module for CurvaDB Level 1.

This module provides PCA-based dimensionality reduction and normalization
to prepare embeddings for Hilbert curve indexing.
"""

from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
import pickle


class DimensionalityReducer:
    """Reduces embedding dimensionality using PCA and normalizes to integer range.

    The reducer performs two main operations:
    1. PCA to reduce from high-dimensional embeddings to n_components dimensions
    2. Min-max normalization to map values to [0, max_val] integer range

    Args:
        n_components: Target number of dimensions after reduction.
        max_val: Maximum integer value for normalization (e.g., 2^16 - 1 = 65535).

    Example:
        >>> reducer = DimensionalityReducer(n_components=8, max_val=65535)
        >>> # Fit on training data
        >>> embeddings = np.random.randn(100, 384)
        >>> reducer.fit(embeddings)
        >>> # Transform new data
        >>> new_emb = np.random.randn(1, 384)
        >>> reduced = reducer.transform(new_emb)
        >>> reduced.shape
        (1, 8)
        >>> reduced.dtype
        dtype('int64')
    """

    def __init__(self, n_components: int = 8, max_val: int = 65535) -> None:
        """Initialize the dimensionality reducer."""
        self.n_components = n_components
        self.max_val = max_val
        self.pca: Optional[PCA] = None
        self.fitted = False
        # Store min/max values for each dimension after PCA
        self.dim_mins: Optional[np.ndarray] = None
        self.dim_maxs: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "DimensionalityReducer":
        """Fit the PCA model on embeddings and compute normalization parameters.

        Args:
            embeddings: Array of shape (n_samples, n_features) to fit on.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If embeddings have fewer samples than n_components.
        """
        if embeddings.shape[0] < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} samples to fit PCA, "
                f"got {embeddings.shape[0]}"
            )

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        reduced = self.pca.fit_transform(embeddings)

        # Compute min/max for each dimension
        self.dim_mins = reduced.min(axis=0)
        self.dim_maxs = reduced.max(axis=0)

        self.fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to reduced and normalized integer coordinates.

        Args:
            embeddings: Array of shape (n_samples, n_features) to transform.

        Returns:
            Array of shape (n_samples, n_components) with integer values in [0, max_val].

        Raises:
            RuntimeError: If reducer has not been fitted yet.
        """
        if not self.fitted or self.pca is None:
            raise RuntimeError("Reducer must be fitted before transform. Call fit() first.")

        # Apply PCA
        reduced = self.pca.transform(embeddings)

        # Normalize each dimension to [0, max_val]
        normalized = np.zeros_like(reduced, dtype=np.int64)
        for i in range(self.n_components):
            dim_min = self.dim_mins[i]
            dim_max = self.dim_maxs[i]
            dim_range = dim_max - dim_min

            if dim_range < 1e-10:  # Avoid division by zero
                normalized[:, i] = self.max_val // 2
            else:
                # Scale to [0, max_val]
                scaled = ((reduced[:, i] - dim_min) / dim_range) * self.max_val
                normalized[:, i] = np.clip(scaled, 0, self.max_val).astype(np.int64)

        return normalized

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform embeddings in one step.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            Transformed array of shape (n_samples, n_components).
        """
        self.fit(embeddings)
        return self.transform(embeddings)

    def save(self, filepath: str) -> None:
        """Save the fitted reducer to disk.

        Args:
            filepath: Path to save the reducer state.

        Raises:
            RuntimeError: If reducer has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted reducer. Call fit() first.")

        state = {
            "n_components": self.n_components,
            "max_val": self.max_val,
            "pca": self.pca,
            "dim_mins": self.dim_mins,
            "dim_maxs": self.dim_maxs,
            "fitted": self.fitted
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str) -> "DimensionalityReducer":
        """Load a fitted reducer from disk.

        Args:
            filepath: Path to load the reducer state from.

        Returns:
            Loaded DimensionalityReducer instance.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        reducer = cls(
            n_components=state["n_components"],
            max_val=state["max_val"]
        )
        reducer.pca = state["pca"]
        reducer.dim_mins = state["dim_mins"]
        reducer.dim_maxs = state["dim_maxs"]
        reducer.fitted = state["fitted"]

        return reducer

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get the variance explained by each principal component.

        Returns:
            Array of explained variance ratios, or None if not fitted.
        """
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def __repr__(self) -> str:
        """String representation of the reducer."""
        status = "fitted" if self.fitted else "not fitted"
        return f"DimensionalityReducer(n_components={self.n_components}, status={status})"
