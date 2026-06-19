"""Tabular reporting utilities for fuzzy-partition experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import relative_error_reduction


def _accuracy_to_percent(value: float) -> float:
    """Convert an accuracy value to a percentage if it is in unit scale."""
    number = float(value)
    return number * 100.0 if 0.0 <= number <= 1.0 else number


def _error_from_accuracy_percent(accuracy_percent: float) -> float:
    """Compute error percentage from accuracy percentage."""
    return 100.0 - float(accuracy_percent)


def _dataframe_to_markdown(table: pd.DataFrame, index: bool = False) -> str:
    """Convert a DataFrame to a simple Markdown table."""
    display = table if index else table.reset_index(drop=True)
    columns = list(display.columns)
    rows = []
    rows.append("| " + " | ".join(str(col) for col in columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in display.iterrows():
        values = ["" if pd.isna(row[col]) else str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def save_table(
    table: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> None:
    """Save a table as CSV, Excel, Markdown, or JSON according to suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(path, index=index)
    elif suffix in {".xlsx", ".xls"}:
        table.to_excel(path, index=index)
    elif suffix in {".md", ".markdown"}:
        path.write_text(_dataframe_to_markdown(table, index=index), encoding="utf-8")
    elif suffix == ".json":
        table.to_json(path, orient="records", indent=2)
    else:
        raise ValueError("output_path must end with .csv, .xlsx, .md, or .json.")


def load_result_table(path: str | Path) -> pd.DataFrame:
    """Load a result table from CSV, Excel, or JSON."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError("path must end with .csv, .xlsx, .xls, or .json.")


def local_result_row(
    family: str,
    degree: int | str | None,
    truncation: str | None,
    accuracy: float,
    cross_entropy: float | None = None,
    k: int | None = None,
    fuzzifier: float | None = None,
    beta: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one row for the local-classification result table."""
    row: dict[str, Any] = {
        "family": family,
        "degree": degree,
        "truncation": truncation,
        "k": k,
        "fuzzifier": fuzzifier,
        "beta": beta,
        "accuracy": float(accuracy),
        "accuracy_percent": _accuracy_to_percent(float(accuracy)),
        "cross_entropy": None if cross_entropy is None else float(cross_entropy),
    }
    if extra:
        row.update(dict(extra))
    return row


def make_local_results_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a local-classification summary table."""
    table = pd.DataFrame(list(rows))
    preferred = [
        "family",
        "degree",
        "truncation",
        "k",
        "fuzzifier",
        "beta",
        "accuracy",
        "accuracy_percent",
        "cross_entropy",
    ]
    ordered = [col for col in preferred if col in table.columns]
    ordered += [col for col in table.columns if col not in ordered]
    return table.loc[:, ordered]


def classifier_comparison_table(
    classifier_names: Sequence[str],
    original_accuracy: Sequence[float],
    incorporated_accuracy: Sequence[float],
) -> pd.DataFrame:
    """Create an original-versus-fuzzy-incorporated accuracy table."""
    if not (len(classifier_names) == len(original_accuracy) == len(incorporated_accuracy)):
        raise ValueError("classifier names and accuracy sequences must have equal length.")

    rows: list[dict[str, Any]] = []
    for name, original, incorporated in zip(classifier_names, original_accuracy, incorporated_accuracy):
        original_percent = _accuracy_to_percent(float(original))
        incorporated_percent = _accuracy_to_percent(float(incorporated))
        rows.append(
            {
                "classifier": name,
                "original_accuracy": original_percent,
                "fuzzy_incorporated_accuracy": incorporated_percent,
                "gain": incorporated_percent - original_percent,
            }
        )
    return pd.DataFrame(rows)


def error_reduction_table(
    classifier_names: Sequence[str],
    original_accuracy: Sequence[float],
    incorporated_accuracy: Sequence[float],
) -> pd.DataFrame:
    """Create an error-reduction table from matched classifier accuracies."""
    if not (len(classifier_names) == len(original_accuracy) == len(incorporated_accuracy)):
        raise ValueError("classifier names and accuracy sequences must have equal length.")

    rows: list[dict[str, Any]] = []
    for name, original, incorporated in zip(classifier_names, original_accuracy, incorporated_accuracy):
        original_percent = _accuracy_to_percent(float(original))
        incorporated_percent = _accuracy_to_percent(float(incorporated))
        original_error = _error_from_accuracy_percent(original_percent)
        incorporated_error = _error_from_accuracy_percent(incorporated_percent)
        reduction = relative_error_reduction(
            baseline_accuracy=original_percent / 100.0,
            improved_accuracy=incorporated_percent / 100.0,
        )
        rows.append(
            {
                "classifier": name,
                "original_error": original_error,
                "fuzzy_incorporated_error": incorporated_error,
                "error_reduction_percent": reduction * 100.0,
            }
        )
    return pd.DataFrame(rows)


def ablation_table(
    representations: Sequence[str],
    descriptions: Sequence[str],
    accuracy: Sequence[float],
) -> pd.DataFrame:
    """Create an ablation table for incorporated feature blocks."""
    if not (len(representations) == len(descriptions) == len(accuracy)):
        raise ValueError("representations, descriptions, and accuracy must have equal length.")
    rows = []
    for representation, description, acc in zip(representations, descriptions, accuracy):
        rows.append(
            {
                "input_representation": representation,
                "main_information": description,
                "accuracy": _accuracy_to_percent(float(acc)),
            }
        )
    return pd.DataFrame(rows)


def truncation_summary_table(
    truncation_names: Sequence[str],
    meanings: Sequence[str],
    accuracy: Sequence[float],
) -> pd.DataFrame:
    """Create a truncation-method summary table."""
    if not (len(truncation_names) == len(meanings) == len(accuracy)):
        raise ValueError("truncation_names, meanings, and accuracy must have equal length.")
    rows = []
    for name, meaning, acc in zip(truncation_names, meanings, accuracy):
        rows.append(
            {
                "truncation": name,
                "main_stability_meaning": meaning,
                "accuracy": _accuracy_to_percent(float(acc)),
            }
        )
    return pd.DataFrame(rows)


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a result object or mapping to a flat dictionary."""
    if isinstance(result, Mapping):
        return dict(result)
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    raise TypeError("result must be a mapping, dataclass instance, or object with __dict__.")


def collect_result_records(results: Iterable[Any]) -> pd.DataFrame:
    """Collect result objects into a DataFrame."""
    return pd.DataFrame([result_to_dict(result) for result in results])


def write_report_bundle(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    formats: Sequence[str] = ("csv", "md"),
) -> None:
    """Write multiple result tables to an output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for table_name, table in tables.items():
        safe_name = table_name.strip().lower().replace(" ", "_").replace("/", "_")
        for extension in formats:
            save_table(table, directory / f"{safe_name}.{extension.lstrip('.')}", index=False)
