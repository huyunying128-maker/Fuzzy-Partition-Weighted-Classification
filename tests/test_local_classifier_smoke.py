"""Smoke tests for the local logit classifier."""

from __future__ import annotations

import numpy as np

from fpwc import PartitionWeightedLocalLogitClassifier


def test_partition_weighted_local_logit_classifier_smoke() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.2],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    classifier = PartitionWeightedLocalLogitClassifier(
        n_centers=2,
        fuzzifier=2.0,
        beta=2.0,
        degree=1,
        truncation="dtd",
        tolerance=0.0,
        max_partition_iter=3,
        random_state=0,
        polynomial_interaction_mode="powers",
    )
    classifier.fit(x, y)

    probabilities = classifier.predict_proba(x)
    predictions = classifier.predict(x)
    details = classifier.predict_with_details(x)

    assert probabilities.shape == (6, 2)
    assert predictions.shape == (6,)
    assert details.local_logits.shape[:2] == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6))
