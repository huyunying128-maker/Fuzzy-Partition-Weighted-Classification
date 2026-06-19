"""Evaluation metrics for classification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationReport:
    """Summary of classification performance."""

    accuracy: float
    error_rate: float
    cross_entropy: float | None = None


def _as_label_vector(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate a one-dimensional label vector."""
    result = np.asarray(labels)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return result


def _as_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Validate a two-dimensional class-probability matrix."""
    result = np.asarray(probabilities, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("probabilities must have at least one row and one column.")
    if not np.all(np.isfinite(result)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(result < 0.0):
        raise ValueError("probabilities must be nonnegative.")
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax probabilities from logits."""
    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.ndim != 2:
        raise ValueError("logits must be a two-dimensional array.")
    if logits_array.shape[0] == 0 or logits_array.shape[1] == 0:
        raise ValueError("logits must have at least one row and one column.")
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("logits must contain only finite values.")

    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Convert integer labels to a one-hot matrix."""
    label_array = _as_label_vector(labels, "labels")
    if not np.issubdtype(label_array.dtype, np.integer):
        if np.all(label_array == label_array.astype(np.int64)):
            label_array = label_array.astype(np.int64)
        else:
            raise ValueError("labels must contain integer class values.")
    else:
        label_array = label_array.astype(np.int64, copy=False)

    if np.any(label_array < 0):
        raise ValueError("labels must be nonnegative integers.")

    if n_classes is None:
        class_count = int(label_array.max()) + 1
    else:
        class_count = int(n_classes)
        if class_count <= 0:
            raise ValueError("n_classes must be positive.")
        if label_array.max() >= class_count:
            raise ValueError("labels contain a value greater than or equal to n_classes.")

    result = np.zeros((label_array.shape[0], class_count), dtype=np.float64)
    result[np.arange(label_array.shape[0]), label_array] = 1.0
    return result


def cross_entropy_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute multiclass cross entropy from class probabilities."""
    labels = _as_label_vector(y_true, "y_true")
    probs = _as_probability_matrix(probabilities)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError(
            "y_true and probabilities must have the same number of rows: "
            f"got {labels.shape[0]} and {probs.shape[0]}."
        )

    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be a positive finite number.")

    encoded = one_hot(labels, n_classes=probs.shape[1])
    return float(-np.mean(np.sum(encoded * np.log(probs + epsilon_value), axis=1)))


def cross_entropy_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute multiclass cross entropy from logits."""
    return cross_entropy_from_probabilities(y_true, softmax(logits), epsilon=epsilon)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    true_labels = _as_label_vector(y_true, "y_true")
    pred_labels = _as_label_vector(y_pred, "y_pred")
    if true_labels.shape[0] != pred_labels.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same length: "
            f"got {true_labels.shape[0]} and {pred_labels.shape[0]}."
        )
    return float(np.mean(true_labels == pred_labels))


def predict_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return predicted class labels from class probabilities."""
    probs = _as_probability_matrix(probabilities)
    return np.argmax(probs, axis=1)


def predict_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return predicted class labels from logits."""
    return predict_from_probabilities(softmax(logits))


def error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification error rate."""
    return 1.0 - accuracy_score(y_true, y_pred)


def relative_error_reduction(
    baseline_accuracy: float,
    improved_accuracy: float,
) -> float:
    """Compute relative error-rate reduction from two accuracies."""
    baseline = float(baseline_accuracy)
    improved = float(improved_accuracy)
    if not np.isfinite(baseline) or not np.isfinite(improved):
        raise ValueError("accuracies must be finite numbers.")
    if baseline < 0.0 or baseline > 1.0 or improved < 0.0 or improved > 1.0:
        raise ValueError("accuracies must be between 0 and 1.")

    baseline_error = 1.0 - baseline
    improved_error = 1.0 - improved
    if baseline_error <= 0.0:
        return 0.0
    return float((baseline_error - improved_error) / baseline_error)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    epsilon: float = 1.0e-12,
) -> ClassificationReport:
    """Create a compact classification performance report."""
    accuracy = accuracy_score(y_true, y_pred)
    ce_value = None
    if probabilities is not None:
        ce_value = cross_entropy_from_probabilities(y_true, probabilities, epsilon=epsilon)
    return ClassificationReport(
        accuracy=accuracy,
        error_rate=1.0 - accuracy,
        cross_entropy=ce_value,
    )
