"""Functional PCA module for Level 2.

Applies Functional Principal Component Analysis to curves for dimensionality reduction.
"""

from typing import List, Optional
import numpy as np
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA as skFPCA
import pickle
import os

from .curve_fitter import Curve


class FunctionalPCA:
    """Functional PCA for curve dimensionality reduction.

    Uses scikit-fda to perform FPCA on curves, extracting low-dimensional
    feature vectors that preserve smooth variation patterns.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_basis: int = 20,
        n_sample_points: int = 50
    ):
        """Initialize Functional PCA.

        Args:
            n_components: Number of principal components to keep
            n_basis: Number of basis functions for functional representation
            n_sample_points: Number of points to sample from each curve
        """
        self.n_components = n_components
        self.n_basis = n_basis
        self.n_sample_points = n_sample_points
        self.fpca = None
        self.fitted = False

        # Store grid points for evaluation
        self.grid_points = np.linspace(0, 1, n_sample_points)

    def _curves_to_fdatagrid(self, curves: List[Curve]) -> FDataGrid:
        """Convert list of curves to FDataGrid format for scikit-fda.

        Args:
            curves: List of Curve objects

        Returns:
            FDataGrid object
        """
        # Sample all curves at same grid points
        sampled_curves = []
        for curve in curves:
            points = curve.sample_uniform(self.n_sample_points)
            # Flatten multivariate curve to 1D for FPCA compatibility
            # (n_points, n_dims) -> (n_points * n_dims,)
            flattened = points.flatten()
            sampled_curves.append(flattened)

        # Stack into shape (n_curves, n_points * n_dims)
        data_matrix = np.array(sampled_curves)

        # Update grid points for flattened representation
        n_total_points = data_matrix.shape[1]
        flat_grid_points = np.linspace(0, 1, n_total_points)

        # Create FDataGrid with 1D function values
        fd = FDataGrid(
            data_matrix=data_matrix,
            grid_points=flat_grid_points
        )

        return fd

    def fit(self, curves: List[Curve]):
        """Fit FPCA on corpus of curves.

        Args:
            curves: List of Curve objects to fit on
        """
        if len(curves) < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} curves to fit FPCA, "
                f"got {len(curves)}"
            )

        # Convert to FDataGrid
        fd = self._curves_to_fdatagrid(curves)

        # Fit FPCA
        self.fpca = skFPCA(n_components=self.n_components)
        self.fpca.fit(fd)
        self.fitted = True

    def transform(self, curves: List[Curve]) -> np.ndarray:
        """Project curves to FPCA space.

        Args:
            curves: List of Curve objects

        Returns:
            FPCA scores, shape (n_curves, n_components)
        """
        if not self.fitted:
            raise RuntimeError("FPCA must be fitted before transform")

        # Convert to FDataGrid
        fd = self._curves_to_fdatagrid(curves)

        # Transform
        scores = self.fpca.transform(fd)

        return scores

    def fit_transform(self, curves: List[Curve]) -> np.ndarray:
        """Fit FPCA and transform in one step.

        Args:
            curves: List of Curve objects

        Returns:
            FPCA scores, shape (n_curves, n_components)
        """
        self.fit(curves)
        return self.transform(curves)

    def inverse_transform(self, scores: np.ndarray) -> List[np.ndarray]:
        """Reconstruct curves from FPCA scores.

        Args:
            scores: FPCA scores, shape (n_curves, n_components)

        Returns:
            List of reconstructed curve points
        """
        if not self.fitted:
            raise RuntimeError("FPCA must be fitted before inverse_transform")

        # Inverse transform
        fd_reconstructed = self.fpca.inverse_transform(scores)

        # Extract data matrix
        reconstructed_curves = fd_reconstructed.data_matrix

        return [curve_data for curve_data in reconstructed_curves]

    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios
        """
        if not self.fitted:
            raise RuntimeError("FPCA must be fitted first")

        return self.fpca.explained_variance_ratio_

    def save(self, filepath: str):
        """Save fitted FPCA model.

        Args:
            filepath: Path to save model
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted FPCA")

        model_data = {
            'fpca': self.fpca,
            'n_components': self.n_components,
            'n_basis': self.n_basis,
            'n_sample_points': self.n_sample_points,
            'grid_points': self.grid_points,
            'fitted': self.fitted
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'FunctionalPCA':
        """Load fitted FPCA model.

        Args:
            filepath: Path to load model from

        Returns:
            Fitted FunctionalPCA object
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        instance = cls(
            n_components=model_data['n_components'],
            n_basis=model_data['n_basis'],
            n_sample_points=model_data['n_sample_points']
        )

        # Restore state
        instance.fpca = model_data['fpca']
        instance.grid_points = model_data['grid_points']
        instance.fitted = model_data['fitted']

        return instance

    def get_component_curves(self) -> List[np.ndarray]:
        """Get the principal component curves.

        Returns:
            List of component curves as arrays
        """
        if not self.fitted:
            raise RuntimeError("FPCA must be fitted first")

        # Get components from fitted model
        components = self.fpca.components_

        return [comp.data_matrix.squeeze() for comp in components]


# Helper function for quick FPCA
def fit_fpca(
    curves: List[Curve],
    n_components: int = 10
) -> FunctionalPCA:
    """Quick helper to fit FPCA on curves.

    Args:
        curves: List of Curve objects
        n_components: Number of components

    Returns:
        Fitted FunctionalPCA object
    """
    fpca = FunctionalPCA(n_components=n_components)
    fpca.fit(curves)
    return fpca
