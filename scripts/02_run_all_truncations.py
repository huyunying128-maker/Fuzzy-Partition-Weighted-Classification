#!/usr/bin/env python
"""Run local-classifier experiments for multiple truncation criteria."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from fpwc.io_utils import ensure_dir, make_run_dir, save_csv, save_json  # noqa: E402
from fpwc.local_classifier import PartitionWeightedLocalLogitClassifier  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402
from fpwc.report_tables import truncation_summary_table, save_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/debug_all_truncations.yaml",
        help="Path to a YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save test predictions for each truncation run.",
    )
    return parser.parse_args()


def _build_logistic_model(logistic_config: dict[str, Any]) -> LogisticRegression:
    """Create the logistic estimator used by each local component."""
    return LogisticRegression(
        C=float(logistic_config.get("C", 1.0)),
        solver=str(logistic_config.get("solver", "lbfgs")),
        max_iter=int(logistic_config.get("max_iter", 1000)),
        fit_intercept=bool(logistic_config.get("fit_intercept", False)),
    )


def _report_to_dict(report) -> dict[str, float | None]:
    """Convert a classification report to a serializable dictionary."""
    return {
        "accuracy": float(report.accuracy),
        "error_rate": float(report.error_rate),
        "cross_entropy": None if report.cross_entropy is None else float(report.cross_entropy),
    }


def _get_required_truncations(config_values: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the configured truncation runs."""
    truncations = config_values.get("truncations")
    if not isinstance(truncations, list) or not truncations:
        raise ValueError("configuration must contain a non-empty 'truncations' list.")
    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(truncations):
        if not isinstance(item, dict):
            raise ValueError(f"truncations[{index}] must be a mapping.")
        if "name" not in item:
            raise ValueError(f"truncations[{index}] is missing the 'name' field.")
        cleaned.append(dict(item))
    return cleaned


def _run_single_truncation(
    truncation_config: dict[str, Any],
    base_model_config: dict[str, Any],
    logistic_config: dict[str, Any],
    x_train,
    y_train,
    x_test,
    y_test,
    random_state: int,
    output_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    """Fit and evaluate one truncation configuration."""
    name = str(truncation_config["name"])
    display_name = str(truncation_config.get("display_name", name))
    run_dir = ensure_dir(output_dir / name)

    model = PartitionWeightedLocalLogitClassifier(
        n_centers=int(truncation_config.get("n_centers", base_model_config.get("n_centers"))),
        fuzzifier=float(truncation_config.get("fuzzifier", base_model_config.get("fuzzifier", 2.0))),
        beta=float(truncation_config.get("beta", base_model_config.get("beta", 2.0))),
        degree=int(truncation_config.get("degree", base_model_config.get("degree", 1))),
        truncation=name,
        tolerance=float(truncation_config.get("tolerance", base_model_config.get("tolerance", 1.0e-6))),
        max_partition_iter=int(
            truncation_config.get("max_partition_iter", base_model_config.get("max_partition_iter", 100))
        ),
        random_state=random_state,
        partition_kind=str(truncation_config.get("partition_kind", base_model_config.get("partition_kind", "fuzzy"))),
        polynomial_interaction_mode=str(
            truncation_config.get(
                "polynomial_interaction_mode",
                base_model_config.get("polynomial_interaction_mode", "full"),
            )
        ),
        logistic_model=_build_logistic_model(logistic_config),
        epsilon=float(truncation_config.get("epsilon", base_model_config.get("epsilon", 1.0e-12))),
    )

    model.fit(x_train, y_train)
    test_details = model.predict_with_details(x_test)
    train_report = model.evaluate(x_train, y_train)
    test_report = model.evaluate(x_test, y_test)

    row = {
        "truncation": name,
        "display_name": display_name,
        "meaning": truncation_config.get("meaning"),
        "n_centers": int(model.n_centers),
        "fuzzifier": float(model.fuzzifier),
        "beta": float(model.beta),
        "degree": int(model.degree),
        "partition_kind": model.partition_kind,
        "train_accuracy": float(train_report.accuracy),
        "train_cross_entropy": None if train_report.cross_entropy is None else float(train_report.cross_entropy),
        "test_accuracy": float(test_report.accuracy),
        "test_cross_entropy": None if test_report.cross_entropy is None else float(test_report.cross_entropy),
        "test_error_rate": float(test_report.error_rate),
        "partition_iterations": int(model.partition_.n_iter),
        "partition_converged": bool(model.partition_.converged),
    }

    save_json(row, run_dir / "results.json")
    save_csv(model.partition_.history, run_dir / "partition_history.csv")

    if save_predictions:
        predictions = pd.DataFrame(
            {
                "sample_index": range(test_details.labels.shape[0]),
                "y_true": y_test,
                "y_pred": test_details.labels,
            }
        )
        save_csv(predictions, run_dir / "test_predictions.csv")

    return row


def main() -> None:
    """Run every truncation setting in the configuration file."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()
    seed = int(config.get("experiment.random_state", 0))
    set_random_seed(seed)

    run_name = args.run_name or str(config.get("experiment.name", "all_truncations"))
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

    truncation_runs = _get_required_truncations(cfg)
    rows: list[dict[str, Any]] = []
    for item in truncation_runs:
        print(f"Running truncation: {item['name']}")
        row = _run_single_truncation(
            truncation_config=item,
            base_model_config=dict(cfg.get("model", {})),
            logistic_config=dict(cfg.get("logistic", {})),
            x_train=split.x_train,
            y_train=split.y_train,
            x_test=split.x_test,
            y_test=split.y_test,
            random_state=seed,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
        )
        rows.append(row)
        print(f"  test accuracy: {row['test_accuracy']:.6f}")

    raw_table = pd.DataFrame(rows)
    save_csv(raw_table, output_dir / "truncation_results.csv")
    save_json({"results": rows}, output_dir / "truncation_results.json")

    display_table = truncation_summary_table(
        truncation_names=[str(row.get("display_name", row["truncation"])) for row in rows],
        meanings=["" if row.get("meaning") is None else str(row.get("meaning")) for row in rows],
        accuracy=[float(row["test_accuracy"]) for row in rows],
    )
    save_table(display_table, output_dir / "table_truncation_summary.csv", index=False)
    save_table(display_table, output_dir / "table_truncation_summary.md", index=False)

    print("All truncation experiments finished")
    print(f"summary: {output_dir / 'truncation_results.csv'}")


if __name__ == "__main__":
    main()
