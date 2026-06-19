#!/usr/bin/env python
"""Create MNIST centroid and membership-prototype visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _add_src_to_path() -> None:
    """Add the local src directory when the package is not installed."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from fpwc.config import load_config, save_config  # noqa: E402
from fpwc.data_mnist import class_centroids, load_mnist_train_split  # noqa: E402
from fpwc.distances import beta_distance_matrix  # noqa: E402
from fpwc.feature_builder import membership_weighted_prototype  # noqa: E402
from fpwc.io_utils import make_run_dir, save_csv, save_json, save_numpy  # noqa: E402
from fpwc.memberships import fuzzy_membership_from_distances  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402
from fpwc.visualization import (  # noqa: E402
    plot_centroid_prototypes,
    plot_digit_standardization_example,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/mnist_centroid_visualization.yaml",
        help="Path to a YAML or JSON visualization configuration.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    return parser.parse_args()


def _select_digit_sample(
    x: np.ndarray,
    y: np.ndarray,
    digit_label: int,
    sample_rank: int,
) -> tuple[int, np.ndarray, int]:
    """Return one sample with the requested label."""
    labels = np.asarray(y)
    matches = np.flatnonzero(labels == int(digit_label))
    if matches.size == 0:
        raise ValueError(f"No sample with label {digit_label} was found.")

    rank = int(sample_rank)
    if rank < 0:
        raise ValueError("sample_rank must be nonnegative.")
    if rank >= matches.size:
        raise ValueError(
            f"sample_rank={rank} is out of range for label {digit_label}; "
            f"only {matches.size} matching samples are available."
        )

    index = int(matches[rank])
    return index, np.asarray(x[index], dtype=np.float64), int(labels[index])


def _blend_images(input_image: np.ndarray, prototype_image: np.ndarray, alpha: float) -> np.ndarray:
    """Blend an input image with its membership-weighted prototype."""
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or alpha_value < 0.0 or alpha_value > 1.0:
        raise ValueError("blend_alpha must be between 0 and 1.")
    return (1.0 - alpha_value) * input_image + alpha_value * prototype_image


def _membership_table(membership: np.ndarray) -> pd.DataFrame:
    """Convert a single membership vector to a table."""
    vector = np.asarray(membership, dtype=np.float64).reshape(-1)
    return pd.DataFrame(
        {
            "class_label": list(range(vector.shape[0])),
            "membership": vector,
        }
    )


def main() -> None:
    """Run the configured visualization workflow."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()

    seed = int(config.get("experiment.random_state", 0))
    set_random_seed(seed)

    run_name = args.run_name or str(config.get("experiment.name", "centroid_visualization"))
    output_dir = make_run_dir(config.get("experiment.output_dir", "outputs"), run_name)
    save_config(cfg, output_dir / "config.yaml")

    split = load_mnist_train_split(
        data_dir=config.get("data.data_dir", "data/mnist"),
        train_size=int(config.get("data.train_size", 48000)),
        test_size=int(config.get("data.test_size", 12000)),
        random_state=seed,
        shuffle=bool(config.get("data.shuffle", True)),
        download=bool(config.get("data.download", True)),
    )

    image_shape_config = config.get("visualization.image_shape", [28, 28])
    image_shape = (int(image_shape_config[0]), int(image_shape_config[1]))
    digit_label = int(config.get("visualization.digit_label", 4))
    sample_rank = int(config.get("visualization.sample_rank", 0))
    fuzzifier = float(config.get("visualization.fuzzifier", 1.3))
    beta = float(config.get("visualization.beta", 1.25))
    blend_alpha = float(config.get("visualization.blend_alpha", 0.5))
    dpi = int(config.get("visualization.dpi", 300))

    centroids = class_centroids(split.x_train, split.y_train, n_classes=split.n_classes)
    sample_index, input_image, true_label = _select_digit_sample(
        split.x_test,
        split.y_test,
        digit_label=digit_label,
        sample_rank=sample_rank,
    )

    distances = beta_distance_matrix(input_image.reshape(1, -1), centroids, beta=beta)
    membership = fuzzy_membership_from_distances(distances, fuzzifier=fuzzifier)
    prototype = membership_weighted_prototype(membership, centroids, fuzzifier=fuzzifier)[0]
    standardized = _blend_images(input_image, prototype, alpha=blend_alpha)

    save_numpy(centroids, output_dir / "class_centroids.npy")
    save_numpy(membership, output_dir / "example_membership.npy")
    save_numpy(prototype, output_dir / "membership_weighted_prototype.npy")
    save_numpy(standardized, output_dir / "standardized_view.npy")
    save_csv(_membership_table(membership), output_dir / "example_membership.csv")

    plot_centroid_prototypes(
        centroids,
        class_labels=list(range(split.n_classes)),
        image_shape=image_shape,
        output_path=output_dir / "centroid_prototypes.png",
        show=False,
    )
    plot_digit_standardization_example(
        input_image=input_image,
        prototype_image=prototype,
        output_image=standardized,
        image_shape=image_shape,
        output_path=output_dir / "digit_standardization_example.png",
        show=False,
        dpi=dpi,
    )

    summary = {
        "sample_index": sample_index,
        "true_label": true_label,
        "requested_digit_label": digit_label,
        "sample_rank": sample_rank,
        "fuzzifier": fuzzifier,
        "beta": beta,
        "blend_alpha": blend_alpha,
        "train_size": int(split.x_train.shape[0]),
        "test_size": int(split.x_test.shape[0]),
        "n_features": int(split.x_train.shape[1]),
        "membership": membership.reshape(-1).tolist(),
        "top_membership_label": int(np.argmax(membership.reshape(-1))),
        "centroid_figure": str(output_dir / "centroid_prototypes.png"),
        "example_figure": str(output_dir / "digit_standardization_example.png"),
    }
    save_json(summary, output_dir / "visualization_summary.json")

    print("Digit centroid visualization finished")
    print(f"sample index: {sample_index}")
    print(f"true label: {true_label}")
    print(f"top membership label: {summary['top_membership_label']}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
