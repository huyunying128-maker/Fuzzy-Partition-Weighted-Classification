"""Visualization utilities for MNIST fuzzy-partition experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _prepare_output_path(output_path: str | Path | None) -> Path | None:
    """Create the parent directory for an output figure path."""
    if output_path is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_image_matrix(image: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a flattened image or image array to a two-dimensional matrix."""
    array = np.asarray(image, dtype=np.float64)
    if array.shape == image_shape:
        return array
    if array.ndim == 1 and array.size == image_shape[0] * image_shape[1]:
        return array.reshape(image_shape)
    raise ValueError(
        "image must either have shape "
        f"{image_shape} or be a flat vector of length {image_shape[0] * image_shape[1]}."
    )


def save_or_show(
    figure: plt.Figure,
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 300,
) -> None:
    """Save and optionally display a Matplotlib figure."""
    path = _prepare_output_path(output_path)
    if path is not None:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_digit_grid(
    images: np.ndarray,
    titles: Sequence[str] | None = None,
    image_shape: tuple[int, int] = (28, 28),
    n_cols: int = 5,
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a grid of digit images.

    Parameters
    ----------
    images:
        Array containing flattened images or image matrices.
    titles:
        Optional title for each image.
    image_shape:
        Height and width used for flattened images.
    n_cols:
        Number of columns in the grid.
    output_path:
        Optional path where the figure is saved.
    show:
        Whether to display the figure interactively.
    dpi:
        Resolution used when saving the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    image_array = np.asarray(images)
    if image_array.ndim == 1:
        image_array = image_array[np.newaxis, :]
    if image_array.shape[0] == 0:
        raise ValueError("images must contain at least one image.")

    n_images = image_array.shape[0]
    if n_cols <= 0:
        raise ValueError("n_cols must be positive.")
    n_rows = int(np.ceil(n_images / n_cols))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))
    axes_array = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index >= n_images:
            continue
        matrix = _as_image_matrix(image_array[index], image_shape=image_shape)
        axis.imshow(matrix, cmap="gray", interpolation="nearest")
        if titles is not None and index < len(titles):
            axis.set_title(str(titles[index]), fontsize=10)

    figure.tight_layout()
    save_or_show(figure, output_path=output_path, show=show, dpi=dpi)
    return figure


def plot_centroid_prototypes(
    centroids: np.ndarray,
    class_labels: Sequence[int | str] | None = None,
    image_shape: tuple[int, int] = (28, 28),
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot class centroid prototypes as a digit grid."""
    centroid_array = np.asarray(centroids, dtype=np.float64)
    if centroid_array.ndim != 2:
        raise ValueError("centroids must be a two-dimensional array.")
    labels = class_labels if class_labels is not None else list(range(centroid_array.shape[0]))
    titles = [f"Class {label}" for label in labels]
    return plot_digit_grid(
        images=centroid_array,
        titles=titles,
        image_shape=image_shape,
        n_cols=min(5, centroid_array.shape[0]),
        output_path=output_path,
        show=show,
    )


def plot_digit_standardization_example(
    input_image: np.ndarray,
    prototype_image: np.ndarray,
    output_image: np.ndarray,
    image_shape: tuple[int, int] = (28, 28),
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot an input digit, fuzzy prototype, and standardized view."""
    images = np.vstack(
        [
            _as_image_matrix(input_image, image_shape=image_shape).reshape(1, -1),
            _as_image_matrix(prototype_image, image_shape=image_shape).reshape(1, -1),
            _as_image_matrix(output_image, image_shape=image_shape).reshape(1, -1),
        ]
    )
    titles = ["Input", "Membership prototype", "Standardized view"]
    return plot_digit_grid(
        images=images,
        titles=titles,
        image_shape=image_shape,
        n_cols=3,
        output_path=output_path,
        show=show,
    )


def plot_classifier_comparison(
    classifier_names: Sequence[str],
    original_accuracy: Sequence[float],
    incorporated_accuracy: Sequence[float],
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:
    """Plot matched classifier accuracies and accuracy gains."""
    names = list(classifier_names)
    original = np.asarray(original_accuracy, dtype=np.float64)
    incorporated = np.asarray(incorporated_accuracy, dtype=np.float64)
    if len(names) == 0:
        raise ValueError("classifier_names must not be empty.")
    if original.shape != incorporated.shape or original.shape[0] != len(names):
        raise ValueError("accuracy arrays must match the number of classifier names.")

    gain = incorporated - original
    x_positions = np.arange(len(names))
    width = 0.35

    figure, axis_left = plt.subplots(figsize=(max(7.0, 1.2 * len(names)), 4.8))
    axis_left.bar(x_positions - width / 2.0, original * 100.0, width, label="Original")
    axis_left.bar(x_positions + width / 2.0, incorporated * 100.0, width, label="Fuzzy incorporated")
    axis_left.set_ylabel("Accuracy (%)")
    axis_left.set_xticks(x_positions)
    axis_left.set_xticklabels(names, rotation=0)
    axis_left.set_ylim(max(0.0, min(original.min(), incorporated.min()) * 100.0 - 2.0), 100.0)
    axis_left.legend(loc="upper left")

    axis_right = axis_left.twinx()
    axis_right.plot(x_positions, gain * 100.0, marker="o", label="Gain")
    axis_right.set_ylabel("Gain (percentage points)")
    axis_right.set_ylim(0.0, max(1.0, gain.max() * 100.0 + 0.5))

    for index, value in enumerate(original):
        axis_left.text(index - width / 2.0, value * 100.0, f"{value * 100.0:.2f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(incorporated):
        axis_left.text(index + width / 2.0, value * 100.0, f"{value * 100.0:.2f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(gain):
        axis_right.text(index, value * 100.0, f"+{value * 100.0:.2f}", ha="center", va="bottom", fontsize=8)

    figure.tight_layout()
    save_or_show(figure, output_path=output_path, show=show, dpi=dpi)
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_labels: Sequence[int | str] | None = None,
    normalize: bool = False,
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 300,
) -> plt.Figure:
    """Plot a confusion matrix."""
    matrix_array = np.asarray(matrix, dtype=np.float64)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array.")

    display_matrix = matrix_array.copy()
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_sums,
            out=np.zeros_like(display_matrix),
            where=row_sums > 0.0,
        )

    labels = list(class_labels) if class_labels is not None else list(range(matrix_array.shape[0]))
    if len(labels) != matrix_array.shape[0]:
        raise ValueError("class_labels must match the size of the confusion matrix.")

    figure, axis = plt.subplots(figsize=(6.0, 5.5))
    image = axis.imshow(display_matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)

    threshold = display_matrix.max() / 2.0 if display_matrix.size > 0 else 0.0
    for row in range(display_matrix.shape[0]):
        for col in range(display_matrix.shape[1]):
            value = display_matrix[row, col]
            text = f"{value:.2f}" if normalize else f"{int(matrix_array[row, col])}"
            axis.text(col, row, text, ha="center", va="center")

    _ = threshold
    figure.tight_layout()
    save_or_show(figure, output_path=output_path, show=show, dpi=dpi)
    return figure
