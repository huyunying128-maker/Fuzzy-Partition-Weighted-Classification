"""Tests for polynomial bases and partition fitting."""

from __future__ import annotations

import numpy as np

from fpwc import (
    PolynomialBasis,
    fit_crisp_partition,
    fit_fuzzy_partition,
    polynomial_basis,
    polynomial_feature_count,
    update_crisp_centers,
    update_fuzzy_centers,
)


def test_polynomial_feature_count_and_basis_values() -> None:
    x = np.array([[2.0, 3.0]])

    assert polynomial_feature_count(2, degree=2, include_bias=True, interaction_mode="full") == 6
    assert polynomial_feature_count(2, degree=2, include_bias=True, interaction_mode="powers") == 5

    powers = polynomial_basis(x, degree=2, include_bias=True, interaction_mode="powers")
    np.testing.assert_allclose(powers, np.array([[1.0, 2.0, 3.0, 4.0, 9.0]]))


def test_polynomial_basis_transformer_tracks_feature_names() -> None:
    transformer = PolynomialBasis(degree=2, include_bias=True, interaction_mode="powers")
    transformer.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert transformer.n_output_features_ == 5
    assert transformer.get_feature_names(["a", "b"]) == ["1", "a", "b", "a^2", "b^2"]


def test_crisp_and_fuzzy_center_updates() -> None:
    x = np.array([[0.0], [2.0], [10.0], [12.0]])
    hard_membership = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    crisp_centers = update_crisp_centers(x, hard_membership)
    fuzzy_centers = update_fuzzy_centers(x, hard_membership, fuzzifier=2.0)

    np.testing.assert_allclose(crisp_centers, np.array([[1.0], [11.0]]))
    np.testing.assert_allclose(fuzzy_centers, np.array([[1.0], [11.0]]))


def test_partition_fitting_returns_valid_shapes() -> None:
    x = np.array([[0.0], [1.0], [9.0], [10.0]])
    initial_centers = np.array([[0.0], [10.0]])

    fuzzy = fit_fuzzy_partition(
        x,
        n_centers=2,
        fuzzifier=2.0,
        beta=2.0,
        truncation="dtd",
        max_iter=5,
        initial_centers=initial_centers,
    )
    crisp = fit_crisp_partition(
        x,
        n_centers=2,
        beta=2.0,
        max_iter=5,
        initial_centers=initial_centers,
    )

    assert fuzzy.centers.shape == (2, 1)
    assert fuzzy.membership.shape == (4, 2)
    assert crisp.membership.shape == (4, 2)
    np.testing.assert_allclose(fuzzy.membership.sum(axis=1), np.ones(4))
    np.testing.assert_allclose(crisp.membership.sum(axis=1), np.ones(4))
