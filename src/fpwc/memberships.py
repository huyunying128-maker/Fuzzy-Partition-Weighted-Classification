"""Membership and aggregation weights for fuzzy partition models."""

from __future__ import annotations

import numpy as np

from .distances import beta_distance_matrix


def _as_distance_matrix(distances: np.ndarray) -> np.ndarray:
    """Validate and return a nonnegative two-dimensional distance matrix."""
    result = np.asarray(distances, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("distances must be a two-dimensional array.")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("distances must have at least one row and one column.")
    if not np.all(np.isfinite(result)):
        raise ValueError("distances must contain only finite values.")
    if np.any(result < 0.0):
        raise ValueError("distances must be nonnegative.")
    return result


def _check_fuzzifier(fuzzifier: float) -> float:
    """Validate the fuzzy membership exponent."""
    value = float(fuzzifier)
    if not np.isfinite(value) or value <= 1.0:
        raise ValueError("fuzzifier must be greater than 1.")
    return value


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize each row of an array so that it sums to one."""
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("row sums must be positive and finite.")
    return values / row_sums


def crisp_membership_from_distances(distances: np.ndarray) -> np.ndarray:
    """Create a hard membership matrix from a distance table.

    Each sample is assigned to the closest center. If several centers share the
    same minimum distance, the first minimum is selected by ``numpy.argmin``.

    Parameters
    ----------
    distances:
        Distance matrix of shape ``(n_samples, n_centers)``.

    Returns
    -------
    numpy.ndarray
        Hard membership matrix of shape ``(n_samples, n_centers)``.
    """
    distance_array = _as_distance_matrix(distances)
    labels = np.argmin(distance_array, axis=1)
    membership = np.zeros_like(distance_array, dtype=np.float64)
    membership[np.arange(distance_array.shape[0]), labels] = 1.0
    return membership


def crisp_membership(
    x: np.ndarray,
    centers: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    """Compute hard memberships from samples and centers."""
    distances = beta_distance_matrix(x=x, centers=centers, beta=beta)
    return crisp_membership_from_distances(distances)


def fuzzy_membership_from_distances(
    distances: np.ndarray,
    fuzzifier: float = 2.0,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    r"""Compute fuzzy memberships from a distance table.

    The membership values satisfy ``membership[i].sum() == 1`` for every
    sample. The implementation uses the algebraically equivalent form

    .. math::
        u_{ij} = \frac{(d_{ij}+\epsilon)^{-2/(f-1)}}
                      {\sum_r (d_{ir}+\epsilon)^{-2/(f-1)}}.

    Exact zero-distance rows are handled explicitly. If a sample coincides
    with one center, its membership is one for that center. If it coincides
    with multiple centers, the membership is divided equally among them.

    Parameters
    ----------
    distances:
        Distance matrix of shape ``(n_samples, n_centers)``.
    fuzzifier:
        Fuzzy membership exponent ``f``. It must be greater than 1.
    epsilon:
        Small positive constant used for numerical stability.

    Returns
    -------
    numpy.ndarray
        Fuzzy membership matrix of shape ``(n_samples, n_centers)``.
    """
    distance_array = _as_distance_matrix(distances)
    fuzzifier_value = _check_fuzzifier(fuzzifier)

    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be a positive finite number.")

    n_samples, n_centers = distance_array.shape
    membership = np.zeros((n_samples, n_centers), dtype=np.float64)

    zero_mask = distance_array <= epsilon_value
    zero_rows = np.any(zero_mask, axis=1)

    if np.any(zero_rows):
        zero_counts = zero_mask[zero_rows].sum(axis=1, keepdims=True)
        membership[zero_rows] = zero_mask[zero_rows] / zero_counts

    nonzero_rows = ~zero_rows
    if np.any(nonzero_rows):
        power = 2.0 / (fuzzifier_value - 1.0)
        stable_distances = distance_array[nonzero_rows] + epsilon_value
        inverse_weights = stable_distances ** (-power)
        membership[nonzero_rows] = _normalize_rows(inverse_weights)

    return membership


def fuzzy_membership(
    x: np.ndarray,
    centers: np.ndarray,
    beta: float = 2.0,
    fuzzifier: float = 2.0,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Compute fuzzy memberships from samples and centers."""
    distances = beta_distance_matrix(x=x, centers=centers, beta=beta)
    return fuzzy_membership_from_distances(
        distances=distances,
        fuzzifier=fuzzifier,
        epsilon=epsilon,
    )


def aggregation_weights(
    membership: np.ndarray,
    fuzzifier: float = 2.0,
) -> np.ndarray:
    r"""Compute normalized fuzzy aggregation weights.

    The returned weights are

    .. math::
        a_{ij} = \frac{u_{ij}^f}{\sum_r u_{ir}^f}.

    Parameters
    ----------
    membership:
        Membership matrix of shape ``(n_samples, n_centers)``.
    fuzzifier:
        Exponent used in the fuzzy aggregation rule.

    Returns
    -------
    numpy.ndarray
        Aggregation weight matrix of shape ``(n_samples, n_centers)``.
    """
    membership_array = np.asarray(membership, dtype=np.float64)
    if membership_array.ndim != 2:
        raise ValueError("membership must be a two-dimensional array.")
    if membership_array.shape[0] == 0 or membership_array.shape[1] == 0:
        raise ValueError("membership must have at least one row and one column.")
    if not np.all(np.isfinite(membership_array)):
        raise ValueError("membership must contain only finite values.")
    if np.any(membership_array < 0.0):
        raise ValueError("membership must be nonnegative.")

    fuzzifier_value = _check_fuzzifier(fuzzifier)
    powered = membership_array**fuzzifier_value
    return _normalize_rows(powered)
