"""MNIST data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MNISTSplit:
    """Flattened MNIST train-test split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int] = (28, 28)
    n_classes: int = 10


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Create or return a NumPy random generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _flatten_images(images: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Flatten image arrays to ``(n_samples, n_features)``."""
    image_array = np.asarray(images)
    if image_array.ndim != 3:
        raise ValueError("images must have shape (n_samples, height, width).")
    if image_array.shape[0] == 0:
        raise ValueError("images must contain at least one sample.")
    return image_array.reshape(image_array.shape[0], -1).astype(dtype, copy=False)


def scale_pixels(x: np.ndarray, max_value: float = 255.0) -> np.ndarray:
    """Scale pixel intensities to the interval ``[0, 1]``."""
    x_array = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x_array)):
        raise ValueError("x must contain only finite values.")

    scale = float(max_value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("max_value must be a positive finite number.")

    return x_array / scale


def train_test_split_arrays(
    x: np.ndarray,
    y: np.ndarray,
    train_size: int,
    test_size: int,
    random_state: int | np.random.Generator | None = 0,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic train-test split from aligned arrays."""
    x_array = np.asarray(x)
    y_array = np.asarray(y)

    if x_array.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "x and y must contain the same number of samples: "
            f"got {x_array.shape[0]} and {y_array.shape[0]}."
        )

    train_count = int(train_size)
    test_count = int(test_size)
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train_size and test_size must be positive.")
    if train_count + test_count > x_array.shape[0]:
        raise ValueError(
            "train_size + test_size cannot exceed the number of samples: "
            f"got {train_count + test_count} and {x_array.shape[0]}."
        )

    indices = np.arange(x_array.shape[0])
    if shuffle:
        generator = _rng(random_state)
        generator.shuffle(indices)

    selected = indices[: train_count + test_count]
    train_indices = selected[:train_count]
    test_indices = selected[train_count:]

    return (
        x_array[train_indices],
        y_array[train_indices],
        x_array[test_indices],
        y_array[test_indices],
    )


def load_mnist_train_split(
    data_dir: str | Path = "data/mnist",
    train_size: int = 48_000,
    test_size: int = 12_000,
    random_state: int | np.random.Generator | None = 0,
    shuffle: bool = True,
    download: bool = True,
) -> MNISTSplit:
    """Load MNIST and split the 60,000 training images.

    Parameters
    ----------
    data_dir:
        Directory where MNIST files are stored.
    train_size:
        Number of samples in the training split.
    test_size:
        Number of samples in the testing split.
    random_state:
        Seed or random generator used when ``shuffle=True``.
    shuffle:
        Shuffle samples before splitting.
    download:
        Allow torchvision to download the dataset if it is not present.

    Returns
    -------
    MNISTSplit
        Flattened pixel arrays scaled to ``[0, 1]`` and integer labels.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for load_mnist_train_split. "
            "Install the project dependencies before loading MNIST."
        ) from exc

    dataset = MNIST(root=str(Path(data_dir)), train=True, download=download)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy().astype(np.int64, copy=False)

    x = scale_pixels(_flatten_images(images))
    x_train, y_train, x_test, y_test = train_test_split_arrays(
        x,
        labels,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return MNISTSplit(
        x_train=x_train,
        y_train=y_train.astype(np.int64, copy=False),
        x_test=x_test,
        y_test=y_test.astype(np.int64, copy=False),
        image_shape=(28, 28),
        n_classes=10,
    )


def class_centroids(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """Compute one centroid per class from labeled samples."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y)

    if x_array.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    if x_array.shape[0] == 0:
        raise ValueError("x and y must contain at least one sample.")

    if n_classes is None:
        class_count = int(np.max(y_array)) + 1
    else:
        class_count = int(n_classes)
        if class_count <= 0:
            raise ValueError("n_classes must be positive.")

    centers = np.zeros((class_count, x_array.shape[1]), dtype=np.float64)
    for label in range(class_count):
        mask = y_array == label
        if not np.any(mask):
            raise ValueError(f"class {label} has no samples.")
        centers[label] = np.mean(x_array[mask], axis=0)

    return centers
