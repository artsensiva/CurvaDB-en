"""Curve distance metrics for Level 2.

Implements various distance measures between parametric curves.
"""

from typing import List, Callable
import numpy as np
try:
    from similaritymeasures import frechet_dist, dtw
    SIMILARITYMEASURES_AVAILABLE = True
except ImportError:
    SIMILARITYMEASURES_AVAILABLE = False
    print("Warning: similaritymeasures not available, using fallback distances")

from .curve_fitter import Curve


def frechet_distance(
    curve1: Curve,
    curve2: Curve,
    n_samples: int = 100
) -> float:
    """Compute discrete Fréchet distance between two curves.

    The Fréchet distance measures similarity between curves, accounting
    for location and ordering of points along the curves.

    Args:
        curve1: First curve
        curve2: Second curve
        n_samples: Number of points to sample from each curve

    Returns:
        Fréchet distance (lower is more similar)
    """
    if not SIMILARITYMEASURES_AVAILABLE:
        # Fallback to L2 distance
        return l2_distance(curve1, curve2, n_samples)

    # Sample both curves uniformly
    points1 = curve1.sample_uniform(n_samples)
    points2 = curve2.sample_uniform(n_samples)

    try:
        # Compute Fréchet distance
        dist = frechet_dist(points1, points2)
        return float(dist)
    except Exception as e:
        # Fallback on error
        print(f"Warning: Fréchet distance failed: {e}, using L2")
        return l2_distance(curve1, curve2, n_samples)


def dtw_distance(
    curve1: Curve,
    curve2: Curve,
    n_samples: int = 100
) -> float:
    """Compute Dynamic Time Warping distance between curves.

    DTW allows for time-warped matching between curves, useful for
    sequences with different pacing.

    Args:
        curve1: First curve
        curve2: Second curve
        n_samples: Number of points to sample

    Returns:
        DTW distance (lower is more similar)
    """
    if not SIMILARITYMEASURES_AVAILABLE:
        # Fallback to L2 distance
        return l2_distance(curve1, curve2, n_samples)

    # Sample both curves
    points1 = curve1.sample_uniform(n_samples)
    points2 = curve2.sample_uniform(n_samples)

    try:
        # Compute DTW distance
        dist, _ = dtw(points1, points2)
        return float(dist)
    except Exception as e:
        # Fallback on error
        print(f"Warning: DTW distance failed: {e}, using L2")
        return l2_distance(curve1, curve2, n_samples)


def l2_distance(
    curve1: Curve,
    curve2: Curve,
    n_samples: int = 100
) -> float:
    """Compute L2 (Euclidean) distance between sampled curves.

    Fast baseline distance metric - computes point-wise Euclidean distance
    between uniformly sampled curves.

    Args:
        curve1: First curve
        curve2: Second curve
        n_samples: Number of points to sample

    Returns:
        L2 distance (lower is more similar)
    """
    # Sample both curves at same points
    points1 = curve1.sample_uniform(n_samples)
    points2 = curve2.sample_uniform(n_samples)

    # Compute point-wise L2 distance
    dist = np.linalg.norm(points1 - points2)

    return float(dist)


def cosine_distance(
    curve1: Curve,
    curve2: Curve,
    n_samples: int = 100
) -> float:
    """Compute cosine distance between flattened curves.

    Args:
        curve1: First curve
        curve2: Second curve
        n_samples: Number of points to sample

    Returns:
        Cosine distance in [0, 2] (lower is more similar)
    """
    # Sample and flatten
    points1 = curve1.sample_uniform(n_samples).flatten()
    points2 = curve2.sample_uniform(n_samples).flatten()

    # Compute cosine similarity
    dot_product = np.dot(points1, points2)
    norm1 = np.linalg.norm(points1)
    norm2 = np.linalg.norm(points2)

    if norm1 == 0 or norm2 == 0:
        return 2.0  # Maximum distance

    cosine_sim = dot_product / (norm1 * norm2)

    # Convert to distance [0, 2]
    cosine_dist = 1.0 - cosine_sim

    return float(cosine_dist)


# Distance function registry
DISTANCE_FUNCTIONS = {
    'frechet': frechet_distance,
    'dtw': dtw_distance,
    'l2': l2_distance,
    'cosine': cosine_distance
}


def compute_distance(
    curve1: Curve,
    curve2: Curve,
    metric: str = 'frechet',
    n_samples: int = 100
) -> float:
    """Compute distance between two curves using specified metric.

    Args:
        curve1: First curve
        curve2: Second curve
        metric: Distance metric ('frechet', 'dtw', 'l2', 'cosine')
        n_samples: Number of points to sample

    Returns:
        Distance value
    """
    if metric not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown distance metric: {metric}. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )

    dist_func = DISTANCE_FUNCTIONS[metric]
    return dist_func(curve1, curve2, n_samples)


def pairwise_distances(
    curves: List[Curve],
    metric: str = 'frechet',
    n_samples: int = 100
) -> np.ndarray:
    """Compute pairwise distance matrix between curves.

    Args:
        curves: List of Curve objects
        metric: Distance metric to use
        n_samples: Number of points to sample

    Returns:
        Distance matrix, shape (n_curves, n_curves)
    """
    n = len(curves)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_distance(curves[i], curves[j], metric, n_samples)
            distances[i, j] = dist
            distances[j, i] = dist  # Symmetric

    return distances


def batch_distances(
    query_curve: Curve,
    candidate_curves: List[Curve],
    metric: str = 'frechet',
    n_samples: int = 100
) -> np.ndarray:
    """Compute distances from query curve to all candidate curves.

    Args:
        query_curve: Query curve
        candidate_curves: List of candidate curves
        metric: Distance metric
        n_samples: Number of points to sample

    Returns:
        Array of distances, shape (n_candidates,)
    """
    distances = np.zeros(len(candidate_curves))

    for i, candidate in enumerate(candidate_curves):
        distances[i] = compute_distance(
            query_curve,
            candidate,
            metric,
            n_samples
        )

    return distances
