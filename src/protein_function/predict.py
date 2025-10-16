"""Command line interface for generating GO term predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import read_fasta_sequences
from .model import ProteinFunctionModel


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .joblib model")
    parser.add_argument("--fasta", required=True, help="FASTA file containing sequences to score")
    parser.add_argument("--output", required=True, help="Destination TSV file for predictions")
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of GO terms to emit per sequence (before min-proba filter)",
    )
    parser.add_argument(
        "--min-proba",
        type=float,
        default=0.0,
        help="Minimum probability to include a GO term in the output",
    )
    return parser.parse_args(args)


def write_predictions(
    output_path: Path,
    sequence_ids: Sequence[str],
    go_terms: Sequence[str],
    probabilities: np.ndarray,
    top_k: int,
    min_proba: float,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, seq_id in enumerate(sequence_ids):
            scores = probabilities[idx]
            top_indices = np.argsort(scores)[::-1][:top_k]
            for term_index in top_indices:
                proba = float(scores[term_index])
                if proba < min_proba:
                    continue
                handle.write(f"{seq_id}\t{go_terms[term_index]}\t{proba:.3f}\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = ProteinFunctionModel.load(args.model)
    sequences = read_fasta_sequences(args.fasta)
    sequence_ids = list(sequences.keys())
    inputs = list(sequences.values())

    probabilities = model.predict_proba(inputs)
    go_terms = model.label_binarizer.classes_
    write_predictions(output_path, sequence_ids, go_terms, probabilities, args.top_k, args.min_proba)
    print(f"Predictions written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
