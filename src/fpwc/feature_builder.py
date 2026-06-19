"""Feature construction for fuzzy partition-weighted classification."""

from __future__ import annotations

import numpy as np


def _as_2d_float_array(array: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite two-dimensional float array."""
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validate_membership(membership: np.ndarray) -> np.ndarray:
    """Validate a nonnegative membership matrix."""
    membership_array = _as_2d_float_array(membership, "membership")
    if np.any(membership_array < 0.0):
        raise ValueError("membership must be nonnegative.")
    return membership_array


def local_weighted_inputs(x: np.ndarray, membership: np.ndarray) -> np.ndarray:
    r"""Construct local gated inputs ``u_ij x_i``.

    Parameters
    ----------
    x:
        Input matrix of shape ``(n_samples, n_features)``.
    membership:
        Membership matrix of shape ``(n_samples, n_centers)``.

    Returns
    -------
    numpy.ndarray
        Three-dimensional array of shape ``(n_samples, n_centers, n_features)``.
        Entry ``result[i, j, :]`` equals ``membership[i, j] * x[i, :]``.
    """
    x_array = _as_2d_float_array(x, "x")
    membership_array = _validate_membership(membership)

    if x_array.shape[0] != membership_array.shape[0]:
        raise ValueError(
            "x and membership must have the same number of rows: "
            f"got {x_array.shape[0]} and {membership_array.shape[0]}."
        )

    return membership_array[:, :, np.newaxis] * x_array[:, np.newaxis, :]


def flatten_local_weighted_inputs(x: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Return local gated inputs as a two-dimensional feature block."""
    local_inputs = local_weighted_inputs(x, membership)
    n_samples = local_inputs.shape[0]
    return local_inputs.reshape(n_samples, -1)


def incorporated_feature_vector(
    x: np.ndarray,
    membership: np.ndarray,
    include_original: bool = True,
    include_membership: bool = True,
    include_local_views: bool = True,
) -> np.ndarray:
    r"""Build the incorporated feature vector used by external classifiers.

    The full representation is

    .. math::
        z_i = [x_i, u_i, u_{i1}x_i, \ldots, u_{ik}x_i].

    Parameters
    ----------
    x:
        Input matrix of shape ``(n_samples, n_features)``.
    membership:
        Membership matrix of shape ``(n_samples, n_centers)``.
    include_original:
        Include the original input block ``x_i``.
    include_membership:
        Include the membership vector ``u_i``.
    include_local_views:
        Include the flattened local gated input block.

    Returns
    -------
    numpy.ndarray
        Feature matrix containing the selected blocks.
    """
    x_array = _as_2d_float_array(x, "x")
    membership_array = _validate_membership(membership)

    if x_array.shape[0] != membership_array.shape[0]:
        raise ValueError(
            "x and membership must have the same number of rows: "
            f"got {x_array.shape[0]} and {membership_array.shape[0]}."
        )

    blocks: list[np.ndarray] = []
    if include_original:
        blocks.append(x_array)
    if include_membership:
        blocks.append(membership_array)
    if include_local_views:
        blocks.append(flatten_local_weighted_inputs(x_array, membership_array))

    if not blocks:
        raise ValueError("at least one feature block must be selected.")

    return np.concatenate(blocks, axis=1)


def membership_weighted_prototype(
    membership: np.ndarray,
    prototypes: np.ndarray,
    fuzzifier: float = 2.0,
) -> np.ndarray:
    r"""Compute membership-weighted prototype reconstructions.

    For prototypes ``m_j``, the reconstruction is

    .. math::
        m_u(x_i) = \frac{\sum_j u_{ij}^{f} m_j}{\sum_j u_{ij}^{f}}.

    Parameters
    ----------
    membership:
        Membership matrix of shape ``(n_samples, n_prototypes)``.
    prototypes:
        Prototype matrix of shape ``(n_prototypes, n_features)``.
    fuzzifier:
        Positive exponent applied to membership weights.

    Returns
    -------
    numpy.ndarray
        Prototype reconstruction matrix of shape ``(n_samples, n_features)``.
    """
    membership_array = _validate_membership(membership)
    prototype_array = _as_2d_float_array(prototypes, "prototypes")

    if membership_array.shape[1] != prototype_array.shape[0]:
        raise ValueError(
            "membership columns must match the number of prototypes: "
            f"got {membership_array.shape[1]} and {prototype_array.shape[0]}."
        )

    fuzzifier_value = float(fuzzifier)
    if not np.isfinite(fuzzifier_value) or fuzzifier_value <= 0.0:
        raise ValueError("fuzzifier must be a positive finite number.")

    weights = membership_array**fuzzifier_value
    denominators = weights.sum(axis=1, keepdims=True)
    if np.any(denominators <= 0.0):
        raise ValueError("each sample must have a positive total prototype weight.")

    return weights @ prototype_array / denominators
