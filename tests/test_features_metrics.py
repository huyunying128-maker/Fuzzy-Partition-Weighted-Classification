"""Tests for feature construction and classification metrics."""

from __future__ import annotations

import numpy as np

from fpwc import (
    accuracy_score,
    cross_entropy_from_logits,
    cross_entropy_from_probabilities,
    error_rate,
    incorporated_feature_vector,
    local_weighted_inputs,
    membership_weighted_prototype,
    one_hot,
    predict_from_probabilities,
    relative_error_reduction,
    softmax,
)


def test_local_weighted_inputs_and_incorporated_vector_shapes() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    membership = np.array([[1.0, 0.0], [0.25, 0.75]])

    local_inputs = local_weighted_inputs(x, membership)
    full_features = incorporated_feature_vector(x, membership)

    assert local_inputs.shape == (2, 2, 2)
    assert full_features.shape == (2, 8)
    np.testing.assert_allclose(local_inputs[1, 0], np.array([0.75, 1.0]))
    np.testing.assert_allclose(local_inputs[1, 1], np.array([2.25, 3.0]))


def test_membership_weighted_prototype_matches_weighted_average() -> None:
    membership = np.array([[1.0, 0.0], [0.5, 0.5]])
    prototypes = np.array([[0.0, 0.0], [2.0, 2.0]])

    reconstruction = membership_weighted_prototype(membership, prototypes, fuzzifier=1.0)

    np.testing.assert_allclose(reconstruction, np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_softmax_and_cross_entropy_are_consistent() -> None:
    logits = np.array([[4.0, 1.0], [0.5, 2.0]])
    labels = np.array([0, 1])
    probabilities = softmax(logits)

    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(
        cross_entropy_from_logits(labels, logits),
        cross_entropy_from_probabilities(labels, probabilities),
    )


def test_label_metrics_and_error_reduction() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.7, 0.3]])

    assert accuracy_score(y_true, y_pred) == 0.75
    assert error_rate(y_true, y_pred) == 0.25
    np.testing.assert_array_equal(predict_from_probabilities(probabilities), np.array([0, 1, 0, 0]))
    np.testing.assert_allclose(one_hot(np.array([0, 2]), n_classes=3), np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert relative_error_reduction(0.8, 0.9) == 0.5
