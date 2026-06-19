# Contributing

Thank you for your interest in improving this repository. Contributions are welcome in the form of bug reports, documentation improvements, tests, and reproducibility updates.

## Development setup

Create a Python environment and install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite before submitting changes:

```bash
python -m pytest -q
```

## Code style

The implementation is organized as a small research package under `src/fpwc/`. Core methods should remain reusable and independent from experiment scripts. Dataset-specific logic should be placed in `scripts/` or clearly named data utilities.

When adding code, prefer:

- clear function names;
- explicit input validation;
- deterministic behavior when a random seed is provided;
- small functions with focused responsibilities;
- tests for numerical formulas and shape assumptions.

## Reproducibility

Experiment scripts should save configuration values, metrics, and generated tables under `outputs/`. Large data files, trained models, and temporary artifacts should not be committed to the repository.

## Reporting issues

When reporting a bug, please include:

- operating system;
- Python version;
- package versions if relevant;
- command used to run the experiment;
- full error message or traceback;
- the configuration file used.
