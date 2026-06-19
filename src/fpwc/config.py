"""Configuration loading and validation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ExperimentConfig:
    """Container for experiment configuration values.

    The configuration is stored as a nested mapping and can be accessed either
    as a dictionary through :attr:`values` or through the :meth:`get` helper.
    """

    values: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value using dot-separated keys.

        Examples
        --------
        ``config.get("partition.k")`` reads ``values["partition"]["k"]``.
        """
        current: Any = self.values
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def require(self, key: str) -> Any:
        """Return a required value and raise ``KeyError`` when it is missing."""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required configuration key: {key}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the configuration values."""
        return deepcopy(self.values)



def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyYAML is required to read YAML configuration files.") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data



def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON configuration file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data



def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration file.

    Parameters
    ----------
    path:
        Path to a ``.yaml``, ``.yml``, or ``.json`` file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        values = _load_yaml(config_path)
    elif suffix == ".json":
        values = _load_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

    return ExperimentConfig(values=values, source=config_path)



def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated with ``updates``."""
    result = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result



def save_config(config: Mapping[str, Any] | ExperimentConfig, path: str | Path) -> Path:
    """Save a configuration as JSON or YAML according to the file suffix."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = config.to_dict() if isinstance(config, ExperimentConfig) else dict(config)

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(values, file, indent=2, sort_keys=True)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("PyYAML is required to write YAML configuration files.") from exc
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(values, file, sort_keys=False)
    else:
        raise ValueError(f"Unsupported configuration format: {output_path.suffix}")

    return output_path
