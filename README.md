# Fuzzy Partition-Weighted Classification

Official implementation for **Technical Report of Incorporating Fuzzy Clustering into Various Machine Learning Methods**.

The repository implements a fuzzy partition-weighted classification framework. The method learns a partition of the input space, computes fuzzy membership values, forms membership-weighted local inputs, fits local logit models, and aggregates local logits using fuzzy weights. The same partition information is also used to construct an incorporated feature vector for external classifiers.

## Method overview

For a sample `x_i`, the model computes distances to centroids and obtains a membership vector

```text
u_i = [u_i1, ..., u_ik].
```

The local input for group `j` is

```text
x_tilde_ij = u_ij * x_i.
```

A local classifier is trained on each local input. The final logits are aggregated with normalized fuzzy weights. For external classifiers, the incorporated representation is

```text
z_i = [x_i, u_i, u_i1 x_i, ..., u_ik x_i].
```

The implementation includes crisp and fuzzy partitioning, distance-table computation with a distance parameter `beta`, fuzzy membership computation with a fuzzifier `f`, truncation criteria, HPD, local logit classification, external classifier comparisons, ablation experiments, and MNIST centroid visualizations.

## Repository structure

```text
.
├── configs/                 # YAML experiment configurations
├── data/                    # Local datasets, not committed
├── docs/                    # Usage and reproducibility notes
├── models/                  # Local trained models, not committed
├── outputs/                 # Local experiment outputs, not committed
├── paper/                   # Paper-related notes and citation material
├── scripts/                 # Command-line experiment scripts
├── src/fpwc/                # Python package source code
├── tests/                   # Pytest test suite
├── assets/                  # Figures for documentation
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

## Installation

Create a Python environment and install the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate fpwc
pip install -e .
```

Run the test suite:

```bash
pytest
```

A quick pipeline check that does not require downloading MNIST is provided by

```bash
python scripts/quick_debug_run.py
```

## MNIST experiments

Prepare the MNIST split:

```bash
python scripts/00_prepare_mnist.py --data_dir data/mnist
```

Run the main fuzzy local HPD experiment:

```bash
python scripts/01_run_local_classifier.py --config configs/mnist_local_hpd.yaml
```

Compare truncation criteria:

```bash
python scripts/02_run_all_truncations.py --config configs/mnist_local_all_truncations.yaml
```

Run original versus fuzzy-incorporated external classifiers:

```bash
python scripts/03_run_external_classifiers.py --config configs/mnist_external_classifiers.yaml
```

Run the feature-block ablation:

```bash
python scripts/04_run_ablation.py --config configs/mnist_ablation.yaml
```

Create digit centroid visualizations:

```bash
python scripts/05_make_digit_centroids.py --config configs/mnist_centroid_visualization.yaml
```

Collect generated CSV and Markdown summary tables:

```bash
python scripts/06_make_summary_tables.py --config configs/summary_tables.yaml
```

## Debug configurations

Small debug configurations are provided for checking the workflow before running the full MNIST experiments:

```bash
python scripts/01_run_local_classifier.py --config configs/debug_small.yaml
python scripts/02_run_all_truncations.py --config configs/debug_all_truncations.yaml
python scripts/03_run_external_classifiers.py --config configs/debug_external_classifiers.yaml
python scripts/04_run_ablation.py --config configs/debug_ablation.yaml
python scripts/05_make_digit_centroids.py --config configs/debug_centroid_visualization.yaml
```

## Outputs

Experiment outputs are written under `outputs/`. Generated datasets, model checkpoints, arrays, and large experiment results are excluded from version control. Summary tables and selected figures may be copied into `assets/` for documentation.

## Citation

Citation metadata is provided in `CITATION.cff`. The formal publication citation can be updated after the final proceedings information is available.

## License

This project is released under the license included in `LICENSE`.
