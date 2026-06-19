"""Distance computations for fuzzy partition models."""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(array: np.ndarray, name: str) -> np.ndarray:
    """Convert an input array to a finite two-dimensional float array."""
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def beta_distance_matrix(
    x: np.ndarray,
    centers: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    r"""Compute the beta-distance table between samples and centers.

    For samples ``x_i`` and centers ``c_j``, the returned matrix contains

    .. math::
        d_{ij,\beta} = \left(\sum_r |x_{ir} - c_{jr}|^\beta\right)^{1/\beta}.

    Parameters
    ----------
    x:
        Array of shape ``(n_samples, n_features)``.
    centers:
        Array of shape ``(n_centers, n_features)``.
    beta:
        Positive distance parameter. ``beta=1`` gives Manhattan distance and
        ``beta=2`` gives Euclidean distance.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape ``(n_samples, n_centers)``.
    """
    x_array = _as_2d_float_array(x, "x")
    center_array = _as_2d_float_array(centers, "centers")

    if x_array.shape[1] != center_array.shape[1]:
        raise ValueError(
            "x and centers must have the same number of features: "
            f"got {x_array.shape[1]} and {center_array.shape[1]}."
        )

    beta_value = float(beta)
    if not np.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("beta must be a positive finite number.")

    diff = np.abs(x_array[:, np.newaxis, :] - center_array[np.newaxis, :, :])

    if beta_value == 1.0:
        distances = np.sum(diff, axis=2)
    elif beta_value == 2.0:
        distances = np.sqrt(np.sum(diff * diff, axis=2))
    else:
        distances = np.sum(diff**beta_value, axis=2) ** (1.0 / beta_value)

    return distances.astype(np.float64, copy=False)
