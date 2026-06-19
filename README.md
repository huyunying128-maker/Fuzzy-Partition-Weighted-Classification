# Fuzzy Partition-Weighted Classification

This repository contains the official implementation for the paper **"Technical Report of Incorporating Fuzzy Clustering into Various Machine Learning Methods"**.

The project studies a fuzzy partition-weighted classification framework. The main model learns a fuzzy partition of the input space, builds membership-weighted local inputs, trains local polynomial/logit classifiers, and aggregates local outputs using normalized fuzzy weights. The same fuzzy information can also be used as an incorporated feature representation for external classifiers such as ANN, CNN, SVM, random forest, and XGBoost.

The implementation is being organized to follow the mathematical formulation in the paper directly. Legacy experimental scripts are not treated as authoritative source code. In particular, PCA-based compression, top-m filtering, memory-saving approximations, or other speed-oriented variants are not part of the default implementation unless they are explicitly added as optional experimental settings later.

## Repository Status

The repository is currently being prepared for public release. The first public version will include:

- reproducible MNIST data preparation;
- crisp and fuzzy partition learning;
- distance-table computation with the distance parameter beta;
- fuzzy membership computation with the fuzzifier f;
- truncation rules including DTD, Shannon entropy, harmonic distance-change control, square probability, and HPD;
- local polynomial/logit classification with fuzzy aggregation;
- fuzzy-incorporated feature construction for external classifiers;
- scripts for reproducing the MNIST tables and figures in the paper.

## Planned Repository Structure

```text
fuzzy-partition-weighted-classification/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/
├── data/
├── src/
│   └── fpwc/
├── scripts/
├── tests/
├── assets/
├── outputs/
└── paper/
```

## Method Overview

For an input vector `x_i`, the method first computes distances from `x_i` to learned centroids and obtains fuzzy memberships `u_i1, ..., u_ik`. The local view for group `j` is

```text
x_tilde_ij = u_ij * x_i
```

A local classifier is trained on the membership-weighted input. The local logits are then aggregated using normalized fuzzy weights. The same partition information can also be used to build an incorporated feature vector

```text
z_i = [x_i, u_i, u_i1 x_i, ..., u_ik x_i]
```

which can be passed to external classifiers.

## Installation

The code will target Python 3.10 or newer. After cloning the repository, install the dependencies with:

```bash
pip install -r requirements.txt
```

For systems where PyTorch installation depends on CUDA or CPU-only settings, install the correct PyTorch build from the official PyTorch installation command first, and then run the command above.

## Reproducing the MNIST Experiments

The final repository will provide script-level entry points similar to the following:

```bash
python scripts/00_prepare_mnist.py --data_dir data/mnist
python scripts/01_run_local_classifier.py --config configs/mnist_local_hpd.yaml
python scripts/02_run_all_truncations.py --config configs/mnist_local_all_truncations.yaml
python scripts/03_run_external_classifiers.py --config configs/mnist_external_classifiers.yaml
python scripts/04_run_ablation.py --config configs/mnist_ablation.yaml
python scripts/05_make_digit_centroids.py --config configs/mnist_centroid_visualization.yaml
python scripts/06_make_summary_tables.py --output_dir outputs/mnist_full_run
```

Large generated files, downloaded datasets, trained models, and experiment outputs are intentionally ignored by Git and should not be committed to the repository.

## Citation

The formal citation will be added after the final publication information is available.

## License

A license file will be added before public release. If no special restriction is required by the institution or coauthors, an MIT License is recommended for research code.
