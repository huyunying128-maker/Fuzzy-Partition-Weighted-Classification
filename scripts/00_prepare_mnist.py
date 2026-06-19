#!/usr/bin/env python
"""Prepare the MNIST split used by the experiment scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Add the local src directory when the package is not installed."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from fpwc.data_mnist import load_mnist_train_split  # noqa: E402
from fpwc.io_utils import make_run_dir, save_json  # noqa: E402
from fpwc.random_utils import set_random_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/mnist", help="MNIST data directory.")
    parser.add_argument("--output_dir", default="outputs", help="Base output directory.")
    parser.add_argument("--train_size", type=int, default=48000, help="Number of training samples.")
    parser.add_argument("--test_size", type=int, default=12000, help="Number of testing samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument("--no_shuffle", action="store_true", help="Use the original MNIST order.")
    parser.add_argument("--no_download", action="store_true", help="Do not download MNIST.")
    return parser.parse_args()


def main() -> None:
    """Load MNIST, create the split, and save a compact summary."""
    args = parse_args()
    set_random_seed(args.seed)

    split = load_mnist_train_split(
        data_dir=args.data_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=not args.no_shuffle,
        download=not args.no_download,
    )

    run_dir = make_run_dir(args.output_dir, "prepare_mnist")
    summary = {
        "data_dir": args.data_dir,
        "train_size": int(split.x_train.shape[0]),
        "test_size": int(split.x_test.shape[0]),
        "n_features": int(split.x_train.shape[1]),
        "n_classes": int(split.n_classes),
        "image_shape": list(split.image_shape),
        "shuffle": not args.no_shuffle,
        "random_state": int(args.seed),
        "x_train_min": float(split.x_train.min()),
        "x_train_max": float(split.x_train.max()),
        "x_test_min": float(split.x_test.min()),
        "x_test_max": float(split.x_test.max()),
    }
    save_json(summary, run_dir / "mnist_split_summary.json")

    print("MNIST split prepared")
    print(f"train: {summary['train_size']} samples")
    print(f"test : {summary['test_size']} samples")
    print(f"features: {summary['n_features']}")
    print(f"summary: {run_dir / 'mnist_split_summary.json'}")


if __name__ == "__main__":
    main()
