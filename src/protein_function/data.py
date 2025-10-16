"""Utilities for loading protein sequences and annotations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


@dataclass
class TrainingExample:
    """Container storing a sequence and its associated GO terms."""

    sequence_id: str
    sequence: str
    go_terms: List[str]


def read_fasta_sequences(path: str | Path) -> Dict[str, str]:
    """Read a FASTA file and return a mapping from sequence id to sequence."""
    sequences: Dict[str, List[str]] = {}
    current_id: str | None = None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = []
            else:
                if current_id is None:
                    raise ValueError("Encountered sequence data before FASTA header.")
                sequences[current_id].append(line.upper())

    return {seq_id: "".join(parts) for seq_id, parts in sequences.items()}


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Load a CSV with columns `sequence_id` and `go_term`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"sequence_id", "go_term"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Annotation file must contain columns {required_columns}, got {set(df.columns)}"
        )
    df = df.dropna(subset=["sequence_id", "go_term"])
    return df


def load_training_data(
    fasta_path: str | Path, annotations_path: str | Path
) -> List[TrainingExample]:
    """Combine FASTA sequences and GO annotations into training examples."""
    sequences = read_fasta_sequences(fasta_path)
    annotations = load_annotations(annotations_path)

    grouped: Dict[str, List[str]] = defaultdict(list)
    for sequence_id, go_term in annotations[["sequence_id", "go_term"]].itertuples(
        index=False
    ):
        grouped[sequence_id].append(str(go_term))

    examples: List[TrainingExample] = []
    missing: List[str] = []
    for sequence_id, go_terms in grouped.items():
        try:
            sequence = sequences[sequence_id]
        except KeyError:
            missing.append(sequence_id)
            continue
        examples.append(TrainingExample(sequence_id, sequence, go_terms))

    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(
            "Annotations referenced sequence ids that are missing from the FASTA file: "
            f"{missing_str}"
        )

    return examples


def iter_sequences_from_examples(examples: Sequence[TrainingExample]) -> Iterable[str]:
    """Yield raw sequences from a list of training examples."""
    for example in examples:
        yield example.sequence


def iter_labels_from_examples(examples: Sequence[TrainingExample]) -> Iterable[List[str]]:
    """Yield GO term lists from a list of training examples."""
    for example in examples:
        yield example.go_terms

