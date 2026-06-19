"""Truncation criteria for partition iterations."""

from __future__ import annotations

import numpy as np


def _as_nonnegative_matrix(array: np.ndarray, name: str) -> np.ndarray:
    """Validate a finite nonnegative two-dimensional array."""
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _check_same_shape(previous: np.ndarray, current: np.ndarray) -> None:
    """Check that two arrays have the same shape."""
    if previous.shape != current.shape:
        raise ValueError(
            "previous and current arrays must have the same shape: "
            f"got {previous.shape} and {current.shape}."
        )


def distance_table_difference(
    previous_distances: np.ndarray,
    current_distances: np.ndarray,
) -> float:
    r"""Compute the mean absolute change in a distance table.

    The criterion is

    .. math::
        D_{DTD}^{(t)} = \frac{1}{nk}\sum_{i,j}
        \left|d_{ij}^{(t)} - d_{ij}^{(t-1)}\right|.

    Parameters
    ----------
    previous_distances:
        Distance table from the previous iteration.
    current_distances:
        Distance table from the current iteration.

    Returns
    -------
    float
        Mean absolute distance-table difference.
    """
    previous = _as_nonnegative_matrix(previous_distances, "previous_distances")
    current = _as_nonnegative_matrix(current_distances, "current_distances")
    _check_same_shape(previous, current)
    return float(np.mean(np.abs(current - previous)))


def shannon_entropy(membership: np.ndarray, epsilon: float = 1.0e-12) -> float:
    r"""Compute the average Shannon entropy of membership rows.

    The criterion is

    .. math::
        H = -\frac{1}{n}\sum_{i,j} u_{ij}\log(u_{ij}+\epsilon).

    Parameters
    ----------
    membership:
        Membership matrix of shape ``(n_samples, n_centers)``.
    epsilon:
        Positive numerical constant used inside the logarithm.

    Returns
    -------
    float
        Average membership entropy.
    """
    membership_array = _as_nonnegative_matrix(membership, "membership")
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be a positive finite number.")

    return float(
        -np.sum(membership_array * np.log(membership_array + epsilon_value))
        / membership_array.shape[0]
    )


def shannon_entropy_change(
    previous_membership: np.ndarray,
    current_membership: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute the absolute change in average Shannon entropy."""
    previous = _as_nonnegative_matrix(previous_membership, "previous_membership")
    current = _as_nonnegative_matrix(current_membership, "current_membership")
    _check_same_shape(previous, current)
    return abs(shannon_entropy(current, epsilon) - shannon_entropy(previous, epsilon))


def harmonic_distance_change(
    previous_distances: np.ndarray,
    current_distances: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    r"""Compute the harmonic mean of distance-table changes.

    The criterion is

    .. math::
        D_{HM}^{(t)} =
        \frac{nk}{\sum_{i,j}(|d_{ij}^{(t)}-d_{ij}^{(t-1)}|+\epsilon)^{-1}}.

    Parameters
    ----------
    previous_distances:
        Distance table from the previous iteration.
    current_distances:
        Distance table from the current iteration.
    epsilon:
        Positive numerical constant used to avoid division by zero.

    Returns
    -------
    float
        Harmonic mean of absolute distance changes.
    """
    previous = _as_nonnegative_matrix(previous_distances, "previous_distances")
    current = _as_nonnegative_matrix(current_distances, "current_distances")
    _check_same_shape(previous, current)

    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be a positive finite number.")

    changes = np.abs(current - previous) + epsilon_value
    return float(changes.size / np.sum(1.0 / changes))


def square_probability_change(
    previous_membership: np.ndarray,
    current_membership: np.ndarray,
) -> float:
    r"""Compute the mean squared change in squared memberships.

    The criterion is

    .. math::
        D_{SP}^{(t)} = \frac{1}{nk}\sum_{i,j}
        \left((u_{ij}^{(t)})^2 - (u_{ij}^{(t-1)})^2\right)^2.

    Parameters
    ----------
    previous_membership:
        Membership matrix from the previous iteration.
    current_membership:
        Membership matrix from the current iteration.

    Returns
    -------
    float
        Mean squared change in squared membership values.
    """
    previous = _as_nonnegative_matrix(previous_membership, "previous_membership")
    current = _as_nonnegative_matrix(current_membership, "current_membership")
    _check_same_shape(previous, current)
    return float(np.mean((current**2 - previous**2) ** 2))
