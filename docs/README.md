# Documentation

This directory contains usage notes for the fuzzy partition-weighted classification codebase.

## Files

- `reproducibility.md`: recommended order for running the MNIST experiments and collecting outputs.

## Main workflow

The repository is organized around configuration files and command-line scripts. Experiment parameters are stored in `configs/`, while the implementation is stored in `src/fpwc/`. Generated results are written to `outputs/` and should not be committed to version control.

A typical workflow is:

1. Install dependencies.
2. Run the tests.
3. Prepare MNIST.
4. Run the local classifier experiment.
5. Run truncation comparisons.
6. Run external classifier comparisons.
7. Run ablation experiments.
8. Generate centroid visualizations.
9. Collect summary tables.

See `reproducibility.md` for command examples.
