"""Run a small synthetic check of the core FPWC pipeline.

This script does not download external data. It creates a small synthetic
classification dataset, fits a fuzzy partition-weighted local classifier, and
prints a compact metric summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fpwc import (
    PartitionWeightedLocalLogitClassifier,
    accuracy_score,
    cross_entropy_from_probabilities,
    set_random_seed,
)


def make_synthetic_data(
    *,
    n_samples: int = 240,
    n_features: int = 4,
    n_classes: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple multiclass dataset with separated Gaussian centers."""
    rng = np.random.default_rng(seed)
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes.")

    centers = rng.normal(loc=0.0, scale=3.0, size=(n_classes, n_features))
    y = np.arange(n_samples) % n_classes
    rng.shuffle(y)
    x = centers[y] + rng.normal(loc=0.0, scale=0.8, size=(n_samples, n_features))
    return x.astype(np.float64), y.astype(int)


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic train/test split."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    n_test = int(round(x.shape[0] * test_fraction))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic FPWC debug check.")
    parser.add_argument("--output_dir", default="outputs/quick_debug", help="Output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--n_samples", type=int, default=240, help="Number of synthetic samples.")
    args = parser.parse_args()

    set_random_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = make_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=args.seed)

    model = PartitionWeightedLocalLogitClassifier(
        n_centers=3,
        fuzzifier=1.5,
        beta=2.0,
        degree=1,
        truncation="hpd",
        tolerance=1e-5,
        max_partition_iter=50,
        random_state=args.seed,
        partition_kind="fuzzy",
        polynomial_interaction_mode="powers",
    )
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)
    pred = model.predict(x_test)
    summary = {
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "cross_entropy": float(cross_entropy_from_probabilities(y_test, prob)),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "quick_debug_summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Saved summary to {output_dir / 'quick_debug_summary.csv'}")


if __name__ == "__main__":
    main()
