"""Level 2: Curve-based semantic search.

This module implements curve-based representation for text sequences,
using B-splines, Functional PCA, and curve distance metrics.
"""

from .curve_fitter import Curve, CurveFitter, fit_curve
from .fpca import FunctionalPCA, fit_fpca
from .distances import (
    frechet_distance,
    dtw_distance,
    l2_distance,
    cosine_distance,
    DISTANCE_FUNCTIONS
)
from .curve_db import CurveDB

__all__ = [
    'Curve',
    'CurveFitter',
    'fit_curve',
    'FunctionalPCA',
    'fit_fpca',
    'frechet_distance',
    'dtw_distance',
    'l2_distance',
    'cosine_distance',
    'DISTANCE_FUNCTIONS',
    'CurveDB',
]
