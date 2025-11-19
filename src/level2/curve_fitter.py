"""Curve fitting module for Level 2.

Converts sequences of token embeddings into parametric curves using B-splines.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.interpolate import splprep, splev


@dataclass
class Curve:
    """Parametric curve representation using B-splines.

    Attributes:
        tck: Tuple of (knots, coefficients, degree) from scipy.interpolate
        dimension: Embedding dimension
        n_points: Number of points used to fit the curve
    """
    tck: tuple  # (t, c, k) from splprep
    dimension: int
    n_points: int

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate curve at parameter values t.

        Args:
            t: Parameter values in [0, 1], shape (n_samples,)

        Returns:
            Points on curve, shape (n_samples, dimension)
        """
        # splev returns tuple of arrays (x, y, ...) for each dimension
        coords = splev(t, self.tck)
        # Stack into shape (n_samples, dimension)
        return np.column_stack(coords)

    def sample_uniform(self, n_samples: int = 100) -> np.ndarray:
        """Sample curve uniformly along parameter space.

        Args:
            n_samples: Number of samples

        Returns:
            Sampled points, shape (n_samples, dimension)
        """
        t = np.linspace(0, 1, n_samples)
        return self.evaluate(t)


class CurveFitter:
    """Fits parametric curves to sequences of embeddings.

    Uses B-spline interpolation to create smooth curves from discrete
    token embedding sequences.
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        degree: int = 3,
        periodic: bool = False
    ):
        """Initialize curve fitter.

        Args:
            smoothing: Smoothing parameter (0 = interpolation, >0 = approximation)
            degree: Spline degree (1=linear, 2=quadratic, 3=cubic)
            periodic: Whether to create periodic (closed) curves
        """
        self.smoothing = smoothing
        self.degree = degree
        self.periodic = periodic

    def tokens_to_curve(
        self,
        token_embeddings: np.ndarray,
        min_points: int = 4
    ) -> Optional[Curve]:
        """Convert token embeddings to parametric curve.

        Args:
            token_embeddings: Token embeddings, shape (n_tokens, embedding_dim)
            min_points: Minimum number of tokens required

        Returns:
            Curve object or None if too few tokens
        """
        n_tokens, embedding_dim = token_embeddings.shape

        # Need at least 2 points for any curve
        if n_tokens < 2:
            return None

        # Adaptive degree selection based on available points
        # splprep requires k < n_tokens
        if n_tokens < self.degree + 1:
            # Use lower degree for few points
            degree = max(1, n_tokens - 1)
        else:
            degree = self.degree

        # Transpose to shape (embedding_dim, n_tokens) for splprep
        # splprep expects each dimension as separate array
        points = token_embeddings.T

        try:
            # Fit B-spline
            # per=1 for periodic, per=0 for non-periodic
            tck, u = splprep(
                points,
                s=self.smoothing,
                k=degree,
                per=1 if self.periodic else 0
            )

            return Curve(
                tck=tck,
                dimension=embedding_dim,
                n_points=n_tokens
            )
        except Exception as e:
            # Spline fitting can fail for degenerate cases
            print(f"Warning: Curve fitting failed: {e}")
            return None

    def fit_batch(
        self,
        embeddings_list: List[np.ndarray],
        show_progress: bool = False
    ) -> List[Optional[Curve]]:
        """Fit curves for multiple embedding sequences.

        Args:
            embeddings_list: List of token embedding arrays
            show_progress: Whether to show progress bar

        Returns:
            List of Curve objects (None for failed fits)
        """
        curves = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(embeddings_list, desc="Fitting curves")
            except ImportError:
                iterator = embeddings_list
        else:
            iterator = embeddings_list

        for embeddings in iterator:
            curve = self.tokens_to_curve(embeddings)
            curves.append(curve)

        return curves

    def curve_to_points(
        self,
        curve: Curve,
        n_samples: int = 50
    ) -> np.ndarray:
        """Sample points from curve for visualization or distance computation.

        Args:
            curve: Curve object
            n_samples: Number of points to sample

        Returns:
            Sampled points, shape (n_samples, dimension)
        """
        return curve.sample_uniform(n_samples)


# Helper function for quick curve fitting
def fit_curve(
    token_embeddings: np.ndarray,
    smoothing: float = 0.0,
    degree: int = 3
) -> Optional[Curve]:
    """Quick helper to fit a single curve.

    Args:
        token_embeddings: Token embeddings, shape (n_tokens, embedding_dim)
        smoothing: Smoothing parameter
        degree: Spline degree

    Returns:
        Curve object or None
    """
    fitter = CurveFitter(smoothing=smoothing, degree=degree)
    return fitter.tokens_to_curve(token_embeddings)
