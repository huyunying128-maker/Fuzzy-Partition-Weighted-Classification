# Models Directory

This directory is reserved for trained model artifacts produced by the experiment scripts.

Typical artifacts may include fitted local classifiers, external classifier checkpoints, partition centers, and serialized evaluation objects. These files are generated during experiments and are not committed to the repository.

Suggested local layout:

```text
models/
├── local/
├── external/
└── partitions/
```

Use the corresponding run directory under `outputs/` to record metrics, configuration files, and table summaries for each experiment.
