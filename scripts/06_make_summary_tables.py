#!/usr/bin/env python
"""Collect experiment outputs and write paper-style summary tables."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _add_src_to_path() -> None:
    """Add the local src directory when the package is not installed."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from fpwc.config import ExperimentConfig, load_config, save_config  # noqa: E402
from fpwc.io_utils import load_json, make_run_dir, save_json  # noqa: E402
from fpwc.report_tables import (  # noqa: E402
    classifier_comparison_table,
    error_reduction_table,
    local_result_row,
    make_local_results_table,
    save_table,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/summary_tables.yaml",
        help="Path to a YAML or JSON summary configuration.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional output subdirectory name.",
    )
    return parser.parse_args()


def _as_path(value: Any) -> Path | None:
    """Convert a path-like configuration value to ``Path``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _dot_get(mapping: dict[str, Any], key: str | None, default: Any = None) -> Any:
    """Read a value from a nested dictionary using a dot-separated key."""
    if key is None:
        return default
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load a saved run configuration when available."""
    for name in ("config.yaml", "config.yml", "config.json"):
        path = run_dir / name
        if path.exists():
            return load_config(path).to_dict()
    return {}


def _formats(config: ExperimentConfig) -> list[str]:
    """Return configured table output formats."""
    values = config.get("table_formats", ["csv", "md"])
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list) or not values:
        raise ValueError("table_formats must be a non-empty list or a string.")
    return [str(item).lstrip(".") for item in values]


def _write_table_bundle(
    table: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    formats: list[str],
) -> list[str]:
    """Write one table in all configured formats."""
    written: list[str] = []
    for extension in formats:
        path = output_dir / f"{base_name}.{extension}"
        save_table(table, path, index=False)
        written.append(str(path))
    return written


def _copy_existing_table(
    source_path: Path,
    output_dir: Path,
    base_name: str,
    formats: list[str],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load an existing CSV table and write it under a standard summary name."""
    if not source_path.exists():
        return None, []
    table = pd.read_csv(source_path)
    return table, _write_table_bundle(table, output_dir, base_name, formats)


def _make_local_table(
    config: ExperimentConfig,
    run_dir: Path,
) -> pd.DataFrame | None:
    """Create a local-classification table from a local experiment run."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None

    result = load_json(results_path)
    run_config = _load_run_config(run_dir)
    local_config = config.get("local_table", {})
    if not isinstance(local_config, dict):
        local_config = {}

    test_result = result.get("test", {})
    if not isinstance(test_result, dict):
        raise ValueError(f"Invalid local results file: {results_path}")

    row = local_result_row(
        family=str(local_config.get("family", "Fuzzy local")),
        degree=_dot_get(run_config, local_config.get("degree_key", "model.degree")),
        truncation=_dot_get(run_config, local_config.get("truncation_key", "model.truncation")),
        accuracy=float(test_result["accuracy"]),
        cross_entropy=test_result.get("cross_entropy"),
        k=_dot_get(run_config, local_config.get("k_key", "model.n_centers")),
        fuzzifier=_dot_get(run_config, local_config.get("fuzzifier_key", "model.fuzzifier")),
        beta=_dot_get(run_config, local_config.get("beta_key", "model.beta")),
        extra={
            "partition_iterations": result.get("partition", {}).get("n_iter")
            if isinstance(result.get("partition"), dict)
            else None,
            "partition_converged": result.get("partition", {}).get("converged")
            if isinstance(result.get("partition"), dict)
            else None,
        },
    )
    return make_local_results_table([row])


def _make_external_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Create external-classifier summary tables from raw results if needed."""
    raw_path = run_dir / "external_classifier_results.csv"
    if not raw_path.exists():
        return {}

    raw = pd.read_csv(raw_path)
    required = {"classifier", "original_accuracy", "fuzzy_incorporated_accuracy"}
    if not required.issubset(raw.columns):
        raise ValueError(f"External classifier results missing required columns: {raw_path}")

    comparison = classifier_comparison_table(
        classifier_names=raw["classifier"].tolist(),
        original_accuracy=raw["original_accuracy"].tolist(),
        incorporated_accuracy=raw["fuzzy_incorporated_accuracy"].tolist(),
    )
    reduction = error_reduction_table(
        classifier_names=raw["classifier"].tolist(),
        original_accuracy=raw["original_accuracy"].tolist(),
        incorporated_accuracy=raw["fuzzy_incorporated_accuracy"].tolist(),
    )
    return {
        "table_classifier_comparison": comparison,
        "table_error_reduction": reduction,
    }


def _collect_tables(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    """Collect all configured result tables."""
    formats = _formats(config)
    source_runs = config.get("source_runs", {})
    if not isinstance(source_runs, dict):
        raise ValueError("source_runs must be a mapping.")

    fail_on_missing = bool(config.get("fail_on_missing", False))
    manifest: dict[str, Any] = {"written_files": [], "missing_sources": []}

    def missing(label: str, path: Path) -> None:
        record = {"source": label, "path": str(path)}
        manifest["missing_sources"].append(record)
        if fail_on_missing:
            raise FileNotFoundError(path)

    local_dir = _as_path(source_runs.get("local_classifier"))
    if local_dir is not None:
        local_table = _make_local_table(config, local_dir)
        if local_table is None:
            missing("local_classifier", local_dir / "results.json")
        else:
            manifest["written_files"].extend(
                _write_table_bundle(local_table, output_dir, "table_local_results", formats)
            )

    truncation_dir = _as_path(source_runs.get("truncation_comparison"))
    if truncation_dir is not None:
        table, written = _copy_existing_table(
            truncation_dir / "table_truncation_summary.csv",
            output_dir,
            "table_truncation_summary",
            formats,
        )
        if table is None:
            missing("truncation_comparison", truncation_dir / "table_truncation_summary.csv")
        else:
            manifest["written_files"].extend(written)
            raw = truncation_dir / "truncation_results.csv"
            if raw.exists():
                destination = output_dir / "truncation_results.csv"
                shutil.copyfile(raw, destination)
                manifest["written_files"].append(str(destination))

    external_dir = _as_path(source_runs.get("external_classifiers"))
    if external_dir is not None:
        existing_comparison, written = _copy_existing_table(
            external_dir / "table_classifier_comparison.csv",
            output_dir,
            "table_classifier_comparison",
            formats,
        )
        if existing_comparison is None:
            generated = _make_external_tables(external_dir)
            if not generated:
                missing("external_classifiers", external_dir / "external_classifier_results.csv")
            else:
                for name, table in generated.items():
                    manifest["written_files"].extend(_write_table_bundle(table, output_dir, name, formats))
        else:
            manifest["written_files"].extend(written)

        existing_reduction, reduction_written = _copy_existing_table(
            external_dir / "table_error_reduction.csv",
            output_dir,
            "table_error_reduction",
            formats,
        )
        if existing_reduction is not None:
            manifest["written_files"].extend(reduction_written)

    ablation_dir = _as_path(source_runs.get("ablation"))
    if ablation_dir is not None:
        table, written = _copy_existing_table(
            ablation_dir / "table_ablation.csv",
            output_dir,
            "table_ablation",
            formats,
        )
        if table is None:
            missing("ablation", ablation_dir / "table_ablation.csv")
        else:
            manifest["written_files"].extend(written)
            raw = ablation_dir / "ablation_results.csv"
            if raw.exists():
                destination = output_dir / "ablation_results.csv"
                shutil.copyfile(raw, destination)
                manifest["written_files"].append(str(destination))

    return manifest


def main() -> None:
    """Run the summary-table collection workflow."""
    args = parse_args()
    config = load_config(args.config)
    cfg = config.to_dict()

    run_name = args.run_name or str(config.get("experiment.name", "summary_tables"))
    output_dir = make_run_dir(config.get("experiment.output_dir", "outputs"), run_name)
    save_config(cfg, output_dir / "config.yaml")

    manifest = _collect_tables(config, output_dir)
    save_json(manifest, output_dir / "summary_manifest.json")

    print("Summary table collection finished")
    print(f"output: {output_dir}")
    if manifest["missing_sources"]:
        print("missing sources:")
        for item in manifest["missing_sources"]:
            print(f"  {item['source']}: {item['path']}")


if __name__ == "__main__":
    main()
