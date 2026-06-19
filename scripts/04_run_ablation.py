#!/usr/bin/env python
"""Run ablation experiments for fuzzy-incorporated feature blocks."""

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
from fpwc.external_classifiers import fit_and_evaluate_external_classifier  # noqa: E402
from fpwc.feature_builder import incorporated_feature_vector  # noqa: E402
from fpwc.io_utils import ensure_dir, make_run_dir, save_csv, save_json, save_numpy  # noqa: E402
from fpwc.memberships import crisp_membership, fuzzy_membership  # noqa: E402
from fpwc.partition import fit_crisp_partition, fit_fuzzy_partition  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402
from fpwc.report_tables import ablation_table, save_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/debug_ablation.yaml",
        help="Path to a YAML or JSON experiment configuration.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    parser.add_argument(
        "--representations",
        nargs="*",
        default=None,
        help="Optional representation keys or labels to run from the configuration.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save test predictions for each representation.",
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


def _selected_representations(
    representation_configs: list[dict[str, Any]],
    selected_names: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter representation configurations by command line selection."""
    if selected_names is None:
        return [cfg for cfg in representation_configs if cfg.get("enabled", True)]

    selected = {name.strip().lower() for name in selected_names}
    output = []
    for item in representation_configs:
        key = str(item.get("key", "")).strip().lower()
        label = str(item.get("label", "")).strip().lower()
        if key in selected or label in selected:
            output.append(item)
    if not output:
        raise ValueError("None of the requested representations were found in the configuration.")
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


def _make_representation(
    x: np.ndarray,
    membership: np.ndarray,
    representation_config: dict[str, Any],
) -> np.ndarray:
    """Construct one configured feature representation."""
    return incorporated_feature_vector(
        x=x,
        membership=membership,
        include_original=bool(representation_config.get("include_original", True)),
        include_membership=bool(representation_config.get("include_membership", False)),
        include_local_views=bool(representation_config.get("include_local_views", False)),
    )


def _save_representation_predictions(
    output_dir: Path,
    key: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    save_probabilities: bool,
) -> None:
    """Save predictions and optional probabilities for one representation."""
    representation_dir = ensure_dir(output_dir / key)
    prediction_table = pd.DataFrame(
        {
            "sample_index": np.arange(y_true.shape[0]),
            "y_true": y_true,
            "y_pred": predictions,
        }
    )
    save_csv(prediction_table, representation_dir / "test_predictions.csv")

    if save_probabilities and probabilities is not None:
        save_numpy(probabilities, representation_dir / "test_probabilities.npy")


def main() -> None:
    """Execute the configured ablation experiment."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()

    seed = int(config.get("experiment.random_state", 0))
    set_random_seed(seed)

    run_name = args.run_name or str(config.get("experiment.name", "ablation"))
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

    representation_configs = cfg.get("representations", [])
    if not isinstance(representation_configs, list) or not representation_configs:
        raise ValueError("configuration must contain a non-empty 'representations' list.")
    representations_to_run = _selected_representations(representation_configs, args.representations)

    classifier_config = _as_dict(cfg.get("classifier"))
    classifier_name = str(classifier_config.get("name", "xgboost"))
    display_name = str(classifier_config.get("display_name", classifier_name))
    classifier_seed = int(classifier_config.get("random_state", seed))
    classifier_params = _as_dict(classifier_config.get("params"))

    rows: list[dict[str, Any]] = []
    for representation in representations_to_run:
        key = str(representation.get("key", representation.get("label", "representation")))
        label = str(representation.get("label", key))
        description = str(representation.get("description", ""))

        print(f"Running representation: {label}")
        x_train_representation = _make_representation(split.x_train, train_membership, representation)
        x_test_representation = _make_representation(split.x_test, test_membership, representation)

        _, result = fit_and_evaluate_external_classifier(
            name=classifier_name,
            x_train=x_train_representation,
            y_train=split.y_train,
            x_test=x_test_representation,
            y_test=split.y_test,
            random_state=classifier_seed,
            **classifier_params,
        )

        row = {
            "representation_key": key,
            "input_representation": label,
            "main_information": description,
            "classifier": display_name,
            "n_features": int(x_train_representation.shape[1]),
            "accuracy": float(result.report.accuracy),
            "error_rate": float(result.report.error_rate),
            "cross_entropy": None
            if result.report.cross_entropy is None
            else float(result.report.cross_entropy),
        }
        rows.append(row)

        if args.save_predictions or args.save_probabilities:
            _save_representation_predictions(
                output_dir=output_dir,
                key=key,
                y_true=split.y_test,
                predictions=result.predictions,
                probabilities=result.probabilities,
                save_probabilities=args.save_probabilities,
            )

        print(f"  accuracy: {result.report.accuracy:.6f}")

    results = pd.DataFrame(rows)
    save_csv(results, output_dir / "ablation_results.csv")
    save_json(
        {
            "partition": partition_info,
            "classifier": {
                "name": classifier_name,
                "display_name": display_name,
                "params": classifier_params,
            },
            "data": {
                "train_size": int(split.x_train.shape[0]),
                "test_size": int(split.x_test.shape[0]),
                "n_original_features": int(split.x_train.shape[1]),
            },
            "results": rows,
        },
        output_dir / "results.json",
    )

    table = ablation_table(
        representations=results["input_representation"].tolist(),
        descriptions=results["main_information"].tolist(),
        accuracy=results["accuracy"].tolist(),
    )
    save_table(table, output_dir / "table_ablation.csv", index=False)
    save_table(table, output_dir / "table_ablation.md", index=False)

    print("Ablation experiment finished")
    print(f"summary: {output_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
