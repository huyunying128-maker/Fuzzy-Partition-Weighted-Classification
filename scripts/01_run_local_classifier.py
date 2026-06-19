#!/usr/bin/env python
"""Run the partition-weighted local logit classifier on MNIST."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def _add_src_to_path() -> None:
    """Add the local src directory when the package is not installed."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from fpwc.config import load_config, save_config  # noqa: E402
from fpwc.data_mnist import load_mnist_train_split  # noqa: E402
from fpwc.io_utils import make_run_dir, save_csv, save_json, save_numpy  # noqa: E402
from fpwc.local_classifier import PartitionWeightedLocalLogitClassifier  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/debug_small.yaml",
        help="Path to a YAML or JSON experiment configuration.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    return parser.parse_args()


def _build_logistic_model(logistic_config: dict) -> LogisticRegression:
    """Create the logistic model used by each local component."""
    return LogisticRegression(
        C=float(logistic_config.get("C", 1.0)),
        solver=str(logistic_config.get("solver", "lbfgs")),
        max_iter=int(logistic_config.get("max_iter", 1000)),
        fit_intercept=bool(logistic_config.get("fit_intercept", False)),
    )


def _report_to_dict(report) -> dict[str, float | None]:
    """Convert a classification report dataclass to a dictionary."""
    return {
        "accuracy": float(report.accuracy),
        "error_rate": float(report.error_rate),
        "cross_entropy": None if report.cross_entropy is None else float(report.cross_entropy),
    }


def main() -> None:
    """Execute the configured local-classifier experiment."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()

    seed = int(config.get("experiment.random_state", 0))
    set_random_seed(seed)

    run_name = args.run_name or str(config.get("experiment.name", "local_classifier"))
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

    logistic_model = _build_logistic_model(cfg.get("logistic", {}))
    model = PartitionWeightedLocalLogitClassifier(
        n_centers=int(config.require("model.n_centers")),
        fuzzifier=float(config.get("model.fuzzifier", 2.0)),
        beta=float(config.get("model.beta", 2.0)),
        degree=int(config.get("model.degree", 1)),
        truncation=str(config.get("model.truncation", "hpd")),
        tolerance=float(config.get("model.tolerance", 1.0e-6)),
        max_partition_iter=int(config.get("model.max_partition_iter", 100)),
        random_state=seed,
        partition_kind=str(config.get("model.partition_kind", "fuzzy")),
        polynomial_interaction_mode=str(config.get("model.polynomial_interaction_mode", "full")),
        logistic_model=logistic_model,
        epsilon=float(config.get("model.epsilon", 1.0e-12)),
    )

    model.fit(split.x_train, split.y_train)
    train_report = model.evaluate(split.x_train, split.y_train)
    test_details = model.predict_with_details(split.x_test)
    test_report = model.evaluate(split.x_test, split.y_test)

    results = {
        "train": _report_to_dict(train_report),
        "test": _report_to_dict(test_report),
        "partition": {
            "n_iter": int(model.partition_.n_iter),
            "converged": bool(model.partition_.converged),
            "n_centers": int(model.partition_.membership.shape[1]),
        },
        "data": {
            "train_size": int(split.x_train.shape[0]),
            "test_size": int(split.x_test.shape[0]),
            "n_features": int(split.x_train.shape[1]),
        },
    }
    save_json(results, output_dir / "results.json")

    predictions = pd.DataFrame(
        {
            "sample_index": range(test_details.labels.shape[0]),
            "y_true": split.y_test,
            "y_pred": test_details.labels,
        }
    )
    save_csv(predictions, output_dir / "test_predictions.csv")
    save_csv(model.partition_.history, output_dir / "partition_history.csv")
    save_numpy(test_details.probabilities, output_dir / "test_probabilities.npy")

    print("Local classifier experiment finished")
    print(f"test accuracy: {test_report.accuracy:.6f}")
    if test_report.cross_entropy is not None:
        print(f"test CE: {test_report.cross_entropy:.6f}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
