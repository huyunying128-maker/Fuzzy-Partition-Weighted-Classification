"""Hereditary partition distance for induced partitions."""

from __future__ import annotations

import numpy as np


def _as_label_vector(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate a one-dimensional label vector."""
    result = np.asarray(labels)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return result


def _choose_two(values: np.ndarray) -> np.ndarray:
    """Compute n choose 2 elementwise."""
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) / 2.0


def contingency_table(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Build the contingency table of two partitions.

    Parameters
    ----------
    labels_a:
        Labels representing the first partition.
    labels_b:
        Labels representing the second partition.

    Returns
    -------
    numpy.ndarray
        Contingency table whose ``(r, s)`` entry is the number of samples that
        belong to cluster ``r`` in the first partition and cluster ``s`` in the
        second partition.
    """
    a = _as_label_vector(labels_a, "labels_a")
    b = _as_label_vector(labels_b, "labels_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "labels_a and labels_b must contain the same number of samples: "
            f"got {a.shape[0]} and {b.shape[0]}."
        )

    _, inverse_a = np.unique(a, return_inverse=True)
    _, inverse_b = np.unique(b, return_inverse=True)

    table = np.zeros((inverse_a.max() + 1, inverse_b.max() + 1), dtype=np.int64)
    np.add.at(table, (inverse_a, inverse_b), 1)
    return table


def hereditary_partition_distance(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    normalize: bool = True,
) -> float:
    r"""Compute the hereditary distance between two finite partitions.

    For a finite partition, the implementation uses pair co-membership counts:

    .. math::
        D_H(\Pi,\Pi') = \frac{1}{2}\{\rho(\Pi)+\rho(\Pi')\}
        - \rho(\Pi\wedge\Pi'),

    where ``rho`` is the sum of within-block sample pairs. The intersection
    refinement ``Pi wedge Pi'`` is represented by the contingency table of the
    two label vectors.

    Parameters
    ----------
    labels_a:
        Labels representing the first partition.
    labels_b:
        Labels representing the second partition.
    normalize:
        If ``True``, divide the distance by the total number of sample pairs.

    Returns
    -------
    float
        Hereditary partition distance. The value is zero when the two
        partitions are identical up to relabeling.
    """
    table = contingency_table(labels_a, labels_b)
    row_counts = table.sum(axis=1)
    col_counts = table.sum(axis=0)

    rho_a = np.sum(_choose_two(row_counts))
    rho_b = np.sum(_choose_two(col_counts))
    rho_meet = np.sum(_choose_two(table))

    distance = 0.5 * (rho_a + rho_b) - rho_meet

    if normalize:
        n_samples = int(table.sum())
        denominator = n_samples * (n_samples - 1) / 2.0
        if denominator > 0.0:
            distance = distance / denominator

    return float(distance)


def labels_from_membership(membership: np.ndarray) -> np.ndarray:
    """Convert a membership matrix to induced partition labels."""
    membership_array = np.asarray(membership, dtype=np.float64)
    if membership_array.ndim != 2:
        raise ValueError("membership must be a two-dimensional array.")
    if membership_array.shape[0] == 0 or membership_array.shape[1] == 0:
        raise ValueError("membership must have at least one row and one column.")
    if not np.all(np.isfinite(membership_array)):
        raise ValueError("membership must contain only finite values.")
    if np.any(membership_array < 0.0):
        raise ValueError("membership must be nonnegative.")
    return np.argmax(membership_array, axis=1)


def hereditary_distance_from_memberships(
    membership_a: np.ndarray,
    membership_b: np.ndarray,
    normalize: bool = True,
) -> float:
    """Compute hereditary distance from two membership matrices."""
    labels_a = labels_from_membership(membership_a)
    labels_b = labels_from_membership(membership_b)
    return hereditary_partition_distance(labels_a, labels_b, normalize=normalize)
