#!/usr/bin/env python
"""Run matched external-classifier comparisons on original and fuzzy features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
from fpwc.external_classifiers import compare_original_and_incorporated  # noqa: E402
from fpwc.feature_builder import incorporated_feature_vector  # noqa: E402
from fpwc.io_utils import ensure_dir, make_run_dir, save_csv, save_json, save_numpy  # noqa: E402
from fpwc.memberships import crisp_membership, fuzzy_membership  # noqa: E402
from fpwc.partition import fit_crisp_partition, fit_fuzzy_partition  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402
from fpwc.report_tables import classifier_comparison_table, error_reduction_table, save_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/debug_external_classifiers.yaml",
        help="Path to a YAML or JSON experiment configuration.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=None,
        help="Optional list of classifier names to run from the configuration.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save test predictions for each classifier and input representation.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Save predicted probabilities when the classifier provides them.",
    )
    return parser.parse_args()


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a mapping value as a dictionary."""
    return dict(value) if isinstance(value, dict) else {}


def _selected_classifiers(
    classifier_configs: list[dict[str, Any]],
    selected_names: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter classifier configurations by command line selection."""
    if selected_names is None:
        return [cfg for cfg in classifier_configs if cfg.get("enabled", True)]

    selected = {name.strip().lower() for name in selected_names}
    output = []
    for item in classifier_configs:
        name = str(item.get("name", "")).strip().lower()
        display_name = str(item.get("display_name", "")).strip().lower()
        if name in selected or display_name in selected:
            output.append(item)
    if not output:
        raise ValueError("None of the requested classifiers were found in the configuration.")
    return output


def _fit_membership_representations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    partition_config: dict[str, Any],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the partition stage and return train/test memberships."""
    kind = str(partition_config.get("kind", "fuzzy")).strip().lower()
    n_centers = int(partition_config.get("n_centers", 10))
    fuzzifier = float(partition_config.get("fuzzifier", 2.0))
    beta = float(partition_config.get("beta", 2.0))
    epsilon = float(partition_config.get("epsilon", 1.0e-12))

    if kind in {"class_centroids", "supervised_centroids"}:
        centers = class_centroids(x_train, y_train, n_classes=n_centers)
        train_membership = fuzzy_membership(
            x=x_train,
            centers=centers,
            beta=beta,
            fuzzifier=fuzzifier,
            epsilon=epsilon,
        )
        test_membership = fuzzy_membership(
            x=x_test,
            centers=centers,
            beta=beta,
            fuzzifier=fuzzifier,
            epsilon=epsilon,
        )
        info = {
            "kind": kind,
            "n_centers": int(centers.shape[0]),
            "fuzzifier": fuzzifier,
            "beta": beta,
            "n_iter": 0,
            "converged": True,
        }
        return train_membership, test_membership, info

    if kind == "fuzzy":
        partition = fit_fuzzy_partition(
            x=x_train,
            n_centers=n_centers,
            fuzzifier=fuzzifier,
            beta=beta,
            truncation=str(partition_config.get("truncation", "hpd")),
            tolerance=float(partition_config.get("tolerance", 1.0e-6)),
            max_iter=int(partition_config.get("max_iter", 100)),
            random_state=random_state,
            epsilon=epsilon,
        )
        test_membership = fuzzy_membership(
            x=x_test,
            centers=partition.centers,
            beta=beta,
            fuzzifier=fuzzifier,
            epsilon=epsilon,
        )
        info = {
            "kind": kind,
            "n_centers": int(partition.centers.shape[0]),
            "fuzzifier": fuzzifier,
            "beta": beta,
            "truncation": str(partition_config.get("truncation", "hpd")),
            "n_iter": int(partition.n_iter),
            "converged": bool(partition.converged),
        }
        return partition.membership, test_membership, info

    if kind == "crisp":
        partition = fit_crisp_partition(
            x=x_train,
            n_centers=n_centers,
            beta=beta,
            truncation=str(partition_config.get("truncation", "hpd")),
            tolerance=float(partition_config.get("tolerance", 1.0e-6)),
            max_iter=int(partition_config.get("max_iter", 100)),
            random_state=random_state,
            epsilon=epsilon,
        )
        test_membership = crisp_membership(x=x_test, centers=partition.centers, beta=beta)
        info = {
            "kind": kind,
            "n_centers": int(partition.centers.shape[0]),
            "beta": beta,
            "truncation": str(partition_config.get("truncation", "hpd")),
            "n_iter": int(partition.n_iter),
            "converged": bool(partition.converged),
        }
        return partition.membership, test_membership, info

    raise ValueError("partition.kind must be fuzzy, crisp, or class_centroids.")


def _report_row(
    display_name: str,
    classifier_name: str,
    original_result,
    incorporated_result,
) -> dict[str, Any]:
    """Create one result row from matched classifier outputs."""
    original_report = original_result.report
    incorporated_report = incorporated_result.report
    return {
        "classifier": display_name,
        "classifier_name": classifier_name,
        "original_accuracy": float(original_report.accuracy),
        "fuzzy_incorporated_accuracy": float(incorporated_report.accuracy),
        "gain": float(incorporated_report.accuracy - original_report.accuracy),
        "original_error_rate": float(original_report.error_rate),
        "fuzzy_incorporated_error_rate": float(incorporated_report.error_rate),
        "original_cross_entropy": None
        if original_report.cross_entropy is None
        else float(original_report.cross_entropy),
        "fuzzy_incorporated_cross_entropy": None
        if incorporated_report.cross_entropy is None
        else float(incorporated_report.cross_entropy),
    }


def _save_predictions(
    output_dir: Path,
    classifier_name: str,
    y_true: np.ndarray,
    original_result,
    incorporated_result,
    save_probabilities: bool,
) -> None:
    """Save predictions and optional probabilities for one classifier."""
    classifier_dir = ensure_dir(output_dir / classifier_name)
    table = pd.DataFrame(
        {
            "sample_index": np.arange(y_true.shape[0]),
            "y_true": y_true,
            "original_pred": original_result.predictions,
            "fuzzy_incorporated_pred": incorporated_result.predictions,
        }
    )
    save_csv(table, classifier_dir / "test_predictions.csv")

    if save_probabilities:
        if original_result.probabilities is not None:
            save_numpy(original_result.probabilities, classifier_dir / "original_probabilities.npy")
        if incorporated_result.probabilities is not None:
            save_numpy(incorporated_result.probabilities, classifier_dir / "fuzzy_incorporated_probabilities.npy")


def main() -> None:
    """Execute the configured external-classifier comparison."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()

    seed = int(config.get("experiment.random_state", 0))
    set_random_seed(seed)

    run_name = args.run_name or str(config.get("experiment.name", "external_classifiers"))
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

    train_membership, test_membership, partition_info = _fit_membership_representations(
        x_train=split.x_train,
        y_train=split.y_train,
        x_test=split.x_test,
        partition_config=_as_dict(cfg.get("partition")),
        random_state=seed,
    )

    representation_config = _as_dict(cfg.get("representation"))
    x_train_incorporated = incorporated_feature_vector(
        split.x_train,
        train_membership,
        include_original=bool(representation_config.get("include_original", True)),
        include_membership=bool(representation_config.get("include_membership", True)),
        include_local_views=bool(representation_config.get("include_local_views", True)),
    )
    x_test_incorporated = incorporated_feature_vector(
        split.x_test,
        test_membership,
        include_original=bool(representation_config.get("include_original", True)),
        include_membership=bool(representation_config.get("include_membership", True)),
        include_local_views=bool(representation_config.get("include_local_views", True)),
    )

    classifier_configs = cfg.get("classifiers", [])
    if not isinstance(classifier_configs, list) or not classifier_configs:
        raise ValueError("configuration must contain a non-empty 'classifiers' list.")
    classifiers_to_run = _selected_classifiers(classifier_configs, args.classifiers)

    rows: list[dict[str, Any]] = []
    for item in classifiers_to_run:
        name = str(item.get("name"))
        display_name = str(item.get("display_name", name))
        params = _as_dict(item.get("params"))
        original_params = _as_dict(item.get("original_params")) or params
        incorporated_params = _as_dict(item.get("incorporated_params")) or params
        classifier_seed = int(item.get("random_state", seed))

        print(f"Running classifier: {display_name}")
        matched = compare_original_and_incorporated(
            name=name,
            x_train_original=split.x_train,
            x_test_original=split.x_test,
            x_train_incorporated=x_train_incorporated,
            x_test_incorporated=x_test_incorporated,
            y_train=split.y_train,
            y_test=split.y_test,
            random_state=classifier_seed,
            original_params=original_params,
            incorporated_params=incorporated_params,
        )
        original_result = matched["original"]
        incorporated_result = matched["incorporated"]
        rows.append(_report_row(display_name, name, original_result, incorporated_result))

        if args.save_predictions or args.save_probabilities:
            _save_predictions(
                output_dir=output_dir,
                classifier_name=name,
                y_true=split.y_test,
                original_result=original_result,
                incorporated_result=incorporated_result,
                save_probabilities=args.save_probabilities,
            )

        print(f"  original accuracy: {original_result.report.accuracy:.6f}")
        print(f"  fuzzy incorporated accuracy: {incorporated_result.report.accuracy:.6f}")

    results_table = pd.DataFrame(rows)
    save_csv(results_table, output_dir / "external_classifier_results.csv")
    save_json(
        {
            "partition": partition_info,
            "data": {
                "train_size": int(split.x_train.shape[0]),
                "test_size": int(split.x_test.shape[0]),
                "n_original_features": int(split.x_train.shape[1]),
                "n_incorporated_features": int(x_train_incorporated.shape[1]),
            },
            "results": rows,
        },
        output_dir / "results.json",
    )

    comparison = classifier_comparison_table(
        classifier_names=results_table["classifier"].tolist(),
        original_accuracy=results_table["original_accuracy"].tolist(),
        incorporated_accuracy=results_table["fuzzy_incorporated_accuracy"].tolist(),
    )
    reduction = error_reduction_table(
        classifier_names=results_table["classifier"].tolist(),
        original_accuracy=results_table["original_accuracy"].tolist(),
        incorporated_accuracy=results_table["fuzzy_incorporated_accuracy"].tolist(),
    )
    save_table(comparison, output_dir / "table_classifier_comparison.csv", index=False)
    save_table(comparison, output_dir / "table_classifier_comparison.md", index=False)
    save_table(reduction, output_dir / "table_error_reduction.csv", index=False)
    save_table(reduction, output_dir / "table_error_reduction.md", index=False)

    print("External classifier comparison finished")
    print(f"summary: {output_dir / 'external_classifier_results.csv'}")


if __name__ == "__main__":
    main()
