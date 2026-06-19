# Reproducibility Guide

This guide describes the recommended order for running the experiments in this repository.

## 1. Install the package

```bash
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest
```

Run the quick synthetic-data check:

```bash
python scripts/quick_debug_run.py
```

## 2. Prepare MNIST

```bash
python scripts/00_prepare_mnist.py --data_dir data/mnist
```

The preparation script checks the sample counts, feature dimension, label values, and pixel range.

## 3. Run the main local model

```bash
python scripts/01_run_local_classifier.py --config configs/mnist_local_hpd.yaml
```

The script writes the main local classifier outputs to the configured output directory, including predictions, probabilities, partition history, and a JSON result summary.

For a faster check, use:

```bash
python scripts/01_run_local_classifier.py --config configs/debug_small.yaml
```

## 4. Compare truncation criteria

```bash
python scripts/02_run_all_truncations.py --config configs/mnist_local_all_truncations.yaml
```

This script compares distance-table difference, harmonic distance-change control, Shannon entropy, square probability, and HPD.

For a faster check, use:

```bash
python scripts/02_run_all_truncations.py --config configs/debug_all_truncations.yaml
```

## 5. Run external classifier comparisons

```bash
python scripts/03_run_external_classifiers.py --config configs/mnist_external_classifiers.yaml
```

This script trains each selected classifier on its original input representation and on the fuzzy-incorporated representation.

For a faster check, use:

```bash
python scripts/03_run_external_classifiers.py --config configs/debug_external_classifiers.yaml
```

## 6. Run ablation experiments

```bash
python scripts/04_run_ablation.py --config configs/mnist_ablation.yaml
```

This script compares feature blocks such as raw input, raw input plus memberships, raw input plus gated views, and the full incorporated representation.

For a faster check, use:

```bash
python scripts/04_run_ablation.py --config configs/debug_ablation.yaml
```

## 7. Create centroid visualizations

```bash
python scripts/05_make_digit_centroids.py --config configs/mnist_centroid_visualization.yaml
```

This script saves class centroids, a centroid prototype figure, a digit standardization example, and the corresponding membership values.

For a faster check, use:

```bash
python scripts/05_make_digit_centroids.py --config configs/debug_centroid_visualization.yaml
```

## 8. Collect summary tables

```bash
python scripts/06_make_summary_tables.py --config configs/summary_tables.yaml
```

The summary script collects generated CSV and JSON files from experiment output folders and writes paper-style tables in CSV and Markdown formats.

## Notes on generated files

The following paths are intended for local generated files and are excluded from version control:

- `data/mnist/`
- `outputs/`
- `models/`
- large arrays such as `.npy` and `.npz`
- model checkpoints such as `.pt`, `.pth`, `.pkl`, and `.joblib`
