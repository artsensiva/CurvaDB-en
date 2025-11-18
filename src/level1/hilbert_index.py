"""Hilbert curve indexing module for CurvaDB Level 1.

This module provides space-filling curve indexing using the Hilbert curve
to map multi-dimensional points to 1-dimensional distances while preserving
spatial locality.
"""

from typing import List
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


class HilbertIndexer:
    """Maps N-dimensional points to 1-dimensional Hilbert curve distances.

    The Hilbert curve is a space-filling curve that maps multi-dimensional
    space to a one-dimensional line while preserving locality - points that
    are close in N-D space tend to be close on the 1-D curve.

    Args:
        p: Number of bits per dimension (precision). Values per dimension: 2^p.
            Example: p=16 means each dimension has 65,536 (2^16) discrete values.
        n: Number of dimensions to index.

    Example:
        >>> indexer = HilbertIndexer(p=10, n=3)
        >>> # Map 3D points to 1D distances
        >>> points = [[100, 200, 150], [101, 201, 151]]
        >>> distances = indexer.points_to_distances(points)
        >>> # Nearby points in 3D space have close distances
        >>> abs(distances[0] - distances[1]) < 1000
        True
    """

    def __init__(self, p: int = 16, n: int = 8) -> None:
        """Initialize the Hilbert indexer.

        Args:
            p: Bits per dimension (precision level).
            n: Number of dimensions.

        Raises:
            ValueError: If p or n are invalid (must be positive integers).
        """
        if p <= 0 or n <= 0:
            raise ValueError(f"p and n must be positive integers, got p={p}, n={n}")

        self.p = p
        self.n = n
        self.max_val = (2 ** p) - 1
        self.hilbert = HilbertCurve(p, n)

    def points_to_distances(self, points: List[List[int]]) -> List[int]:
        """Convert N-dimensional points to 1-dimensional Hilbert distances.

        Args:
            points: List of points, where each point is a list of n integers
                   in range [0, 2^p - 1].

        Returns:
            List of Hilbert distances (integers) for each point.

        Example:
            >>> indexer = HilbertIndexer(p=8, n=2)
            >>> points = [[100, 150], [200, 50]]
            >>> distances = indexer.points_to_distances(points)
            >>> len(distances)
            2
        """
        # Validate points
        for i, point in enumerate(points):
            if len(point) != self.n:
                raise ValueError(
                    f"Point {i} has {len(point)} dimensions, expected {self.n}"
                )
            for j, val in enumerate(point):
                if not (0 <= val <= self.max_val):
                    raise ValueError(
                        f"Point {i}, dimension {j}: value {val} out of range "
                        f"[0, {self.max_val}]"
                    )

        # Convert to Hilbert distances
        distances = self.hilbert.distances_from_points(points)
        return distances

    def distances_to_points(self, distances: List[int]) -> List[List[int]]:
        """Convert 1-dimensional Hilbert distances back to N-dimensional points.

        This is mainly useful for debugging and visualization.

        Args:
            distances: List of Hilbert distance integers.

        Returns:
            List of N-dimensional points.

        Example:
            >>> indexer = HilbertIndexer(p=8, n=2)
            >>> distances = [1000, 2000]
            >>> points = indexer.distances_to_points(distances)
            >>> len(points[0])
            2
        """
        points = self.hilbert.points_from_distances(distances)
        return points

    def get_max_distance(self) -> int:
        """Get the maximum possible Hilbert distance for this configuration.

        Returns:
            Maximum distance value (2^(p*n) - 1).
        """
        return (2 ** (self.p * self.n)) - 1

    def validate_point(self, point: List[int]) -> bool:
        """Check if a point is valid for this indexer configuration.

        Args:
            point: N-dimensional point to validate.

        Returns:
            True if point is valid, False otherwise.
        """
        if len(point) != self.n:
            return False
        for val in point:
            if not (0 <= val <= self.max_val):
                return False
        return True

    def __repr__(self) -> str:
        """String representation of the indexer."""
        return (
            f"HilbertIndexer(p={self.p}, n={self.n}, "
            f"values_per_dim={2**self.p}, max_distance={self.get_max_distance()})"
        )
