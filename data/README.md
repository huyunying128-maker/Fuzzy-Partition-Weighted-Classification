# Data Directory

This directory is reserved for datasets used by the experiment scripts.

The MNIST scripts download or read the dataset locally and create the split used in the experiments. Generated dataset files should remain local and are not committed to the repository.

Expected local layout after preparation:

```text
data/
└── mnist/
    ├── raw/
    └── processed/
```

The default MNIST split used by the scripts is:

```text
train samples: 48,000
test samples:  12,000
input dimension: 784
pixel scale: [0, 1]
```

Large data files are ignored by Git. This keeps the repository lightweight and reproducible.
