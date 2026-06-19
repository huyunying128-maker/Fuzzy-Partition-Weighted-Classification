"""Input and output helpers for experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd



def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def timestamp_utc() -> str:
    """Return a compact UTC timestamp suitable for file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")



def make_run_dir(base_dir: str | Path, name: str | None = None) -> Path:
    """Create and return a run directory under ``base_dir``."""
    base = ensure_dir(base_dir)
    run_name = name or f"run_{timestamp_utc()}"
    return ensure_dir(base / run_name)



def save_json(data: Mapping[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    """Save a mapping to a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, sort_keys=True, default=_json_default)
    return output_path



def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file as a dictionary."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {input_path}")
    return data



def save_csv(table: pd.DataFrame | Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Save tabular records to a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(list(table))
    frame.to_csv(output_path, index=False)
    return output_path



def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file as a ``pandas.DataFrame``."""
    return pd.read_csv(path)



def save_numpy(array: np.ndarray, path: str | Path) -> Path:
    """Save an array in NumPy ``.npy`` format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    return output_path



def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy ``.npy`` array."""
    return np.load(path)



def _json_default(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
