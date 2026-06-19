"""Tests for truncation criteria and hereditary partition distance."""

from __future__ import annotations

import numpy as np

from fpwc import (
    distance_table_difference,
    hereditary_distance_from_memberships,
    hereditary_partition_distance,
    harmonic_distance_change,
    shannon_entropy,
    shannon_entropy_change,
    square_probability_change,
)


def test_distance_table_difference_is_mean_absolute_change() -> None:
    previous = np.array([[1.0, 2.0], [3.0, 4.0]])
    current = np.array([[2.0, 2.0], [1.0, 5.0]])

    assert distance_table_difference(previous, current) == 1.0


def test_membership_based_truncation_criteria() -> None:
    one_hot = np.array([[1.0, 0.0], [0.0, 1.0]])
    uniform = np.array([[0.5, 0.5], [0.5, 0.5]])

    assert shannon_entropy(one_hot) < shannon_entropy(uniform)
    assert shannon_entropy_change(one_hot, one_hot) == 0.0
    assert square_probability_change(uniform, uniform) == 0.0


def test_harmonic_distance_change_is_positive_for_changed_tables() -> None:
    previous = np.array([[1.0, 2.0], [3.0, 4.0]])
    current = np.array([[1.5, 2.0], [3.5, 5.0]])

    value = harmonic_distance_change(previous, current)
    assert value > 0.0


def test_hereditary_partition_distance_is_zero_for_relabeling() -> None:
    labels_a = np.array([0, 0, 1, 1])
    labels_b = np.array([5, 5, 9, 9])

    assert hereditary_partition_distance(labels_a, labels_b) == 0.0


def test_hereditary_partition_distance_detects_crossing_partitions() -> None:
    labels_a = np.array([0, 0, 1, 1])
    labels_b = np.array([0, 1, 0, 1])

    assert hereditary_partition_distance(labels_a, labels_b) == 1.0 / 3.0


def test_hereditary_distance_from_memberships_uses_induced_labels() -> None:
    membership_a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    membership_b = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])

    assert hereditary_distance_from_memberships(membership_a, membership_b) == 0.0
