"""Tests for distance tables and membership weights."""

from __future__ import annotations

import numpy as np
import pytest

from fpwc import (
    aggregation_weights,
    beta_distance_matrix,
    crisp_membership_from_distances,
    fuzzy_membership_from_distances,
)


def test_beta_distance_matrix_matches_standard_lp_distances() -> None:
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    centers = np.array([[0.0, 0.0], [3.0, 0.0]])

    euclidean = beta_distance_matrix(x, centers, beta=2.0)
    manhattan = beta_distance_matrix(x, centers, beta=1.0)

    np.testing.assert_allclose(euclidean, np.array([[0.0, 3.0], [5.0, 4.0]]))
    np.testing.assert_allclose(manhattan, np.array([[0.0, 3.0], [7.0, 4.0]]))


def test_beta_distance_matrix_rejects_invalid_beta() -> None:
    x = np.array([[0.0, 1.0]])
    centers = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="beta"):
        beta_distance_matrix(x, centers, beta=0.0)


def test_crisp_membership_assigns_nearest_center() -> None:
    distances = np.array([[0.2, 1.0, 2.0], [5.0, 3.0, 1.0]])
    membership = crisp_membership_from_distances(distances)

    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(membership, expected)


def test_fuzzy_membership_rows_sum_to_one_and_handle_zero_distance() -> None:
    distances = np.array([[0.0, 2.0], [1.0, 3.0]])
    membership = fuzzy_membership_from_distances(distances, fuzzifier=2.0)

    np.testing.assert_allclose(membership.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(membership[0], np.array([1.0, 0.0]))
    assert membership[1, 0] > membership[1, 1]


def test_aggregation_weights_apply_fuzzy_power_and_normalize_rows() -> None:
    membership = np.array([[0.5, 0.5], [0.2, 0.8]])
    weights = aggregation_weights(membership, fuzzifier=2.0)

    np.testing.assert_allclose(weights.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(weights[0], np.array([0.5, 0.5]))
    np.testing.assert_allclose(weights[1], np.array([0.04 / 0.68, 0.64 / 0.68]))
