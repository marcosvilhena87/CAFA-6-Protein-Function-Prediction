"""Command line interface for training the baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .data import (
    iter_labels_from_examples,
    iter_sequences_from_examples,
    load_training_data,
)
from .model import ProteinFunctionModel


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, help="Path to the training FASTA file")
    parser.add_argument(
        "--annotations",
        required=True,
        help="CSV file with columns sequence_id, go_term",
    )
    parser.add_argument(
        "--output", required=True, help="Where to store the trained model (.joblib)"
    )
    parser.add_argument(
        "--penalty",
        default="l2",
        choices=["l1", "l2"],
        help="Penalty type for logistic regression",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    examples = load_training_data(args.fasta, args.annotations)

    model = ProteinFunctionModel(penalty=args.penalty, c=args.c, max_iter=args.max_iter)
    sequences = list(iter_sequences_from_examples(examples))
    labels = list(iter_labels_from_examples(examples))
    model.fit(sequences, labels)
    model.save(args.output)
    print(f"Model saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
