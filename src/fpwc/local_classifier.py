"""Fuzzy partition-weighted local logit classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from .distances import beta_distance_matrix
from .memberships import (
    aggregation_weights,
    crisp_membership_from_distances,
    fuzzy_membership_from_distances,
)
from .metrics import ClassificationReport, classification_report, softmax
from .partition import PartitionResult, fit_crisp_partition, fit_fuzzy_partition
from .polynomial_basis import PolynomialBasis


PartitionKind = Literal["fuzzy", "crisp"]


@dataclass(frozen=True)
class LocalLogitPrediction:
    """Prediction details returned by the local logit classifier."""

    logits: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    membership: np.ndarray
    aggregation: np.ndarray
    local_logits: np.ndarray


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


def _as_label_vector(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a one-dimensional label vector."""
    result = np.asarray(labels)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return result


def _normalize_partition_kind(partition_kind: str) -> PartitionKind:
    """Normalize the partition type name."""
    key = str(partition_kind).lower().strip()
    if key not in {"fuzzy", "crisp"}:
        raise ValueError("partition_kind must be either 'fuzzy' or 'crisp'.")
    return key  # type: ignore[return-value]


def _logistic_logits(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    """Return a two-dimensional logit matrix from a logistic model."""
    scores = model.decision_function(features)
    if scores.ndim == 1:
        return np.column_stack([-scores, scores])
    return np.asarray(scores, dtype=np.float64)


class PartitionWeightedLocalLogitClassifier:
    r"""Local polynomial/logit classifier with fuzzy aggregation.

    The classifier first learns a partition of the input space. For each local
    group ``j``, it trains a multinomial logistic model on the gated input
    ``u_ij x_i`` after polynomial expansion. The final logit is the weighted
    sum of local logits.
    """

    def __init__(
        self,
        n_centers: int,
        fuzzifier: float = 2.0,
        beta: float = 2.0,
        degree: int = 1,
        truncation: str = "hpd",
        tolerance: float = 1.0e-6,
        max_partition_iter: int = 100,
        random_state: int | np.random.Generator | None = 0,
        partition_kind: str = "fuzzy",
        polynomial_interaction_mode: str = "full",
        logistic_model: LogisticRegression | None = None,
        epsilon: float = 1.0e-12,
    ) -> None:
        self.n_centers = int(n_centers)
        self.fuzzifier = float(fuzzifier)
        self.beta = float(beta)
        self.degree = int(degree)
        self.truncation = str(truncation)
        self.tolerance = float(tolerance)
        self.max_partition_iter = int(max_partition_iter)
        self.random_state = random_state
        self.partition_kind = _normalize_partition_kind(partition_kind)
        self.polynomial_interaction_mode = str(polynomial_interaction_mode)
        self.epsilon = float(epsilon)

        if logistic_model is None:
            logistic_model = LogisticRegression(
                C=1.0,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=1000,
            )
        self.logistic_model = logistic_model

        self.partition_: PartitionResult | None = None
        self.bases_: list[PolynomialBasis] | None = None
        self.local_models_: list[LogisticRegression] | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PartitionWeightedLocalLogitClassifier":
        """Fit the partition and the local logit models."""
        x_array = _as_2d_float_array(x, "x")
        y_array = _as_label_vector(y, "y")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError(
                "x and y must contain the same number of samples: "
                f"got {x_array.shape[0]} and {y_array.shape[0]}."
            )
        classes = np.unique(y_array)
        if classes.shape[0] < 2:
            raise ValueError("at least two classes are required for logistic classification.")

        if self.partition_kind == "fuzzy":
            partition = fit_fuzzy_partition(
                x=x_array,
                n_centers=self.n_centers,
                fuzzifier=self.fuzzifier,
                beta=self.beta,
                truncation=self.truncation,
                tolerance=self.tolerance,
                max_iter=self.max_partition_iter,
                random_state=self.random_state,
                epsilon=self.epsilon,
            )
        else:
            partition = fit_crisp_partition(
                x=x_array,
                n_centers=self.n_centers,
                beta=self.beta,
                tolerance=self.tolerance,
                max_iter=self.max_partition_iter,
                random_state=self.random_state,
            )

        bases: list[PolynomialBasis] = []
        local_models: list[LogisticRegression] = []
        membership = partition.membership

        for center_index in range(membership.shape[1]):
            gated_input = membership[:, [center_index]] * x_array
            basis = PolynomialBasis(
                degree=self.degree,
                include_bias=True,
                interaction_mode=self.polynomial_interaction_mode,
            )
            features = basis.fit_transform(gated_input)
            model = clone(self.logistic_model)
            model.fit(features, y_array)
            if not np.array_equal(model.classes_, classes):
                raise RuntimeError("local model classes differ from the global class set.")
            bases.append(basis)
            local_models.append(model)

        self.partition_ = partition
        self.bases_ = bases
        self.local_models_ = local_models
        self.classes_ = classes
        self.n_features_in_ = x_array.shape[1]
        return self

    def _check_is_fitted(self) -> None:
        """Check whether the classifier has been fitted."""
        if (
            self.partition_ is None
            or self.bases_ is None
            or self.local_models_ is None
            or self.classes_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("classifier has not been fitted.")

    def _membership_for_input(self, x: np.ndarray) -> np.ndarray:
        """Compute membership values for new samples."""
        self._check_is_fitted()
        assert self.partition_ is not None
        distances = beta_distance_matrix(x, self.partition_.centers, beta=self.beta)
        if self.partition_kind == "fuzzy":
            return fuzzy_membership_from_distances(
                distances,
                fuzzifier=self.fuzzifier,
                epsilon=self.epsilon,
            )
        return crisp_membership_from_distances(distances)

    def _aggregation_for_membership(self, membership: np.ndarray) -> np.ndarray:
        """Compute aggregation weights from membership values."""
        if self.partition_kind == "fuzzy":
            return aggregation_weights(membership, fuzzifier=self.fuzzifier)
        return membership.astype(np.float64, copy=False)

    def local_logits(self, x: np.ndarray, membership: np.ndarray | None = None) -> np.ndarray:
        """Return local logits for every sample and local group."""
        self._check_is_fitted()
        assert self.bases_ is not None
        assert self.local_models_ is not None
        assert self.classes_ is not None
        assert self.n_features_in_ is not None

        x_array = _as_2d_float_array(x, "x")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(
                "x has a different number of features from the fitted input: "
                f"expected {self.n_features_in_}, got {x_array.shape[1]}."
            )
        if membership is None:
            membership_array = self._membership_for_input(x_array)
        else:
            membership_array = np.asarray(membership, dtype=np.float64)
            if membership_array.shape != (x_array.shape[0], len(self.local_models_)):
                raise ValueError(
                    "membership must have shape "
                    f"{(x_array.shape[0], len(self.local_models_))}, "
                    f"got {membership_array.shape}."
                )

        logits = np.empty(
            (x_array.shape[0], len(self.local_models_), self.classes_.shape[0]),
            dtype=np.float64,
        )
        for center_index, (basis, model) in enumerate(zip(self.bases_, self.local_models_)):
            gated_input = membership_array[:, [center_index]] * x_array
            features = basis.transform(gated_input)
            logits[:, center_index, :] = _logistic_logits(model, features)
        return logits

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Return aggregated logits for each class."""
        x_array = _as_2d_float_array(x, "x")
        membership = self._membership_for_input(x_array)
        aggregation = self._aggregation_for_membership(membership)
        local = self.local_logits(x_array, membership=membership)
        return np.sum(aggregation[:, :, np.newaxis] * local, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for input samples."""
        return softmax(self.decision_function(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        self._check_is_fitted()
        assert self.classes_ is not None
        probabilities = self.predict_proba(x)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_with_details(self, x: np.ndarray) -> LocalLogitPrediction:
        """Return predictions together with membership and local-logit details."""
        self._check_is_fitted()
        assert self.classes_ is not None
        x_array = _as_2d_float_array(x, "x")
        membership = self._membership_for_input(x_array)
        aggregation = self._aggregation_for_membership(membership)
        local = self.local_logits(x_array, membership=membership)
        logits = np.sum(aggregation[:, :, np.newaxis] * local, axis=1)
        probabilities = softmax(logits)
        labels = self.classes_[np.argmax(probabilities, axis=1)]
        return LocalLogitPrediction(
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            membership=membership,
            aggregation=aggregation,
            local_logits=local,
        )

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        report = self.evaluate(x, y)
        return report.accuracy

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> ClassificationReport:
        """Return accuracy, error rate, and cross entropy."""
        details = self.predict_with_details(x)
        return classification_report(
            y_true=y,
            y_pred=details.labels,
            probabilities=details.probabilities,
            epsilon=self.epsilon,
        )
