"""Partition fitting and centroid updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .distances import beta_distance_matrix
from .hpd import hereditary_partition_distance
from .memberships import crisp_membership_from_distances, fuzzy_membership_from_distances
from .truncation import (
    distance_table_difference,
    harmonic_distance_change,
    shannon_entropy_change,
    square_probability_change,
)


@dataclass(frozen=True)
class PartitionResult:
    """Result returned by a partition fitting routine."""

    centers: np.ndarray
    membership: np.ndarray
    labels: np.ndarray
    distances: np.ndarray
    history: list[dict[str, Any]]
    n_iter: int
    converged: bool


def _as_2d_float_array(array: np.ndarray, name: str) -> np.ndarray:
    """Validate a finite two-dimensional float array."""
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


def _check_n_centers(n_centers: int, n_samples: int) -> int:
    """Validate the number of centers."""
    value = int(n_centers)
    if value < 1:
        raise ValueError("n_centers must be at least 1.")
    if value > n_samples:
        raise ValueError("n_centers cannot exceed the number of samples.")
    return value


def initialize_centers(
    x: np.ndarray,
    n_centers: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Select initial centers from the input samples without replacement."""
    x_array = _as_2d_float_array(x, "x")
    k = _check_n_centers(n_centers, x_array.shape[0])

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    indices = rng.choice(x_array.shape[0], size=k, replace=False)
    return x_array[indices].copy()


def update_crisp_centers(
    x: np.ndarray,
    membership: np.ndarray,
    previous_centers: np.ndarray | None = None,
) -> np.ndarray:
    """Update centers by hard membership averages."""
    x_array = _as_2d_float_array(x, "x")
    membership_array = _as_2d_float_array(membership, "membership")
    if x_array.shape[0] != membership_array.shape[0]:
        raise ValueError("x and membership must have the same number of rows.")

    weights = membership_array
    counts = weights.sum(axis=0)
    centers = np.zeros((membership_array.shape[1], x_array.shape[1]), dtype=np.float64)

    nonempty = counts > 0.0
    centers[nonempty] = weights[:, nonempty].T @ x_array / counts[nonempty, None]

    if np.any(~nonempty):
        if previous_centers is None:
            raise ValueError("empty crisp cluster encountered without previous centers.")
        previous = _as_2d_float_array(previous_centers, "previous_centers")
        if previous.shape != centers.shape:
            raise ValueError(
                "previous_centers must have shape "
                f"{centers.shape}, got {previous.shape}."
            )
        centers[~nonempty] = previous[~nonempty]

    return centers


def update_fuzzy_centers(
    x: np.ndarray,
    membership: np.ndarray,
    fuzzifier: float,
) -> np.ndarray:
    r"""Update fuzzy centers with membership powers.

    The update is

    .. math::
        c_j = \frac{\sum_i u_{ij}^{f} x_i}{\sum_i u_{ij}^{f}}.

    Parameters
    ----------
    x:
        Input matrix of shape ``(n_samples, n_features)``.
    membership:
        Membership matrix of shape ``(n_samples, n_centers)``.
    fuzzifier:
        Fuzzy membership exponent.

    Returns
    -------
    numpy.ndarray
        Updated centers of shape ``(n_centers, n_features)``.
    """
    x_array = _as_2d_float_array(x, "x")
    membership_array = _as_2d_float_array(membership, "membership")
    if x_array.shape[0] != membership_array.shape[0]:
        raise ValueError("x and membership must have the same number of rows.")

    fuzzifier_value = float(fuzzifier)
    if not np.isfinite(fuzzifier_value) or fuzzifier_value <= 1.0:
        raise ValueError("fuzzifier must be greater than 1.")

    weights = membership_array**fuzzifier_value
    denominators = weights.sum(axis=0)
    if np.any(denominators <= 0.0):
        raise ValueError("each fuzzy center must have a positive total weight.")

    return weights.T @ x_array / denominators[:, None]


def _truncation_value(
    method: str,
    previous_distances: np.ndarray,
    current_distances: np.ndarray,
    previous_membership: np.ndarray,
    current_membership: np.ndarray,
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    epsilon: float,
) -> float:
    """Evaluate a truncation criterion for two consecutive iterations."""
    method_key = method.lower().replace("-", "_").replace(" ", "_")

    if method_key in {"dtd", "distance", "distance_table_difference"}:
        return distance_table_difference(previous_distances, current_distances)
    if method_key in {"harmonic", "hm", "harmonic_distance_change"}:
        return harmonic_distance_change(previous_distances, current_distances, epsilon=epsilon)
    if method_key in {"shannon", "entropy", "shannon_entropy"}:
        return shannon_entropy_change(previous_membership, current_membership, epsilon=epsilon)
    if method_key in {"square_probability", "sp"}:
        return square_probability_change(previous_membership, current_membership)
    if method_key in {"hpd", "hereditary", "hereditary_partition_distance"}:
        return hereditary_partition_distance(previous_labels, current_labels, normalize=True)

    raise ValueError(
        "unknown truncation method. Expected one of: "
        "dtd, harmonic, shannon, square_probability, hpd."
    )


def fit_fuzzy_partition(
    x: np.ndarray,
    n_centers: int,
    fuzzifier: float = 2.0,
    beta: float = 2.0,
    truncation: str = "hpd",
    tolerance: float = 1.0e-6,
    max_iter: int = 100,
    random_state: int | np.random.Generator | None = 0,
    initial_centers: np.ndarray | None = None,
    epsilon: float = 1.0e-12,
) -> PartitionResult:
    """Fit a fuzzy partition by alternating membership and center updates.

    Parameters
    ----------
    x:
        Input matrix of shape ``(n_samples, n_features)``.
    n_centers:
        Number of partition centers.
    fuzzifier:
        Fuzzy membership exponent.
    beta:
        Distance parameter used in the beta-distance table.
    truncation:
        Truncation criterion used for convergence checking.
    tolerance:
        Convergence threshold for the selected truncation criterion.
    max_iter:
        Maximum number of partition updates.
    random_state:
        Seed or random generator used for center initialization.
    initial_centers:
        Optional initial center matrix.
    epsilon:
        Positive numerical constant used by membership and truncation routines.

    Returns
    -------
    PartitionResult
        Final centers, memberships, induced labels, distances, and iteration
        history.
    """
    x_array = _as_2d_float_array(x, "x")
    k = _check_n_centers(n_centers, x_array.shape[0])

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a nonnegative finite number.")

    if initial_centers is None:
        centers = initialize_centers(x_array, k, random_state=random_state)
    else:
        centers = _as_2d_float_array(initial_centers, "initial_centers").copy()
        if centers.shape != (k, x_array.shape[1]):
            raise ValueError(
                "initial_centers must have shape "
                f"{(k, x_array.shape[1])}, got {centers.shape}."
            )

    history: list[dict[str, Any]] = []
    previous_distances: np.ndarray | None = None
    previous_membership: np.ndarray | None = None
    previous_labels: np.ndarray | None = None
    converged = False

    distances = beta_distance_matrix(x_array, centers, beta=beta)
    membership = fuzzy_membership_from_distances(
        distances,
        fuzzifier=fuzzifier,
        epsilon=epsilon,
    )
    labels = np.argmax(membership, axis=1)

    for iteration in range(1, max_iter + 1):
        centers = update_fuzzy_centers(x_array, membership, fuzzifier=fuzzifier)
        distances = beta_distance_matrix(x_array, centers, beta=beta)
        current_membership = fuzzy_membership_from_distances(
            distances,
            fuzzifier=fuzzifier,
            epsilon=epsilon,
        )
        current_labels = np.argmax(current_membership, axis=1)

        criterion_value = None
        if previous_distances is not None:
            criterion_value = _truncation_value(
                truncation,
                previous_distances=previous_distances,
                current_distances=distances,
                previous_membership=previous_membership,
                current_membership=current_membership,
                previous_labels=previous_labels,
                current_labels=current_labels,
                epsilon=epsilon,
            )
            converged = criterion_value <= tolerance

        history.append(
            {
                "iteration": iteration,
                "truncation": truncation,
                "criterion_value": criterion_value,
            }
        )

        previous_distances = distances
        previous_membership = current_membership
        previous_labels = current_labels
        membership = current_membership
        labels = current_labels

        if converged:
            break

    return PartitionResult(
        centers=centers,
        membership=membership,
        labels=labels,
        distances=distances,
        history=history,
        n_iter=len(history),
        converged=converged,
    )


def fit_crisp_partition(
    x: np.ndarray,
    n_centers: int,
    beta: float = 2.0,
    tolerance: float = 1.0e-6,
    max_iter: int = 100,
    random_state: int | np.random.Generator | None = 0,
    initial_centers: np.ndarray | None = None,
) -> PartitionResult:
    """Fit a crisp partition by alternating assignment and center updates."""
    x_array = _as_2d_float_array(x, "x")
    k = _check_n_centers(n_centers, x_array.shape[0])

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a nonnegative finite number.")

    if initial_centers is None:
        centers = initialize_centers(x_array, k, random_state=random_state)
    else:
        centers = _as_2d_float_array(initial_centers, "initial_centers").copy()
        if centers.shape != (k, x_array.shape[1]):
            raise ValueError(
                "initial_centers must have shape "
                f"{(k, x_array.shape[1])}, got {centers.shape}."
            )

    history: list[dict[str, Any]] = []
    converged = False
    previous_distances: np.ndarray | None = None

    distances = beta_distance_matrix(x_array, centers, beta=beta)
    membership = crisp_membership_from_distances(distances)
    labels = np.argmax(membership, axis=1)

    for iteration in range(1, max_iter + 1):
        centers = update_crisp_centers(x_array, membership, previous_centers=centers)
        distances = beta_distance_matrix(x_array, centers, beta=beta)
        membership = crisp_membership_from_distances(distances)
        labels = np.argmax(membership, axis=1)

        criterion_value = None
        if previous_distances is not None:
            criterion_value = distance_table_difference(previous_distances, distances)
            converged = criterion_value <= tolerance

        history.append(
            {
                "iteration": iteration,
                "truncation": "distance_table_difference",
                "criterion_value": criterion_value,
            }
        )

        previous_distances = distances
        if converged:
            break

    return PartitionResult(
        centers=centers,
        membership=membership,
        labels=labels,
        distances=distances,
        history=history,
        n_iter=len(history),
        converged=converged,
    )
