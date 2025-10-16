"""Feature extraction utilities for protein sequences."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC: set[str] = set("AILMFWYV")
POLAR: set[str] = set("STNQ")
POSITIVELY_CHARGED: set[str] = set("KRH")
NEGATIVELY_CHARGED: set[str] = set("DE")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_composition(sequence: str) -> np.ndarray:
    """Return normalized amino-acid counts for the 20 canonical residues."""
    counts = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    for idx, aa in enumerate(AMINO_ACIDS):
        counts[idx] = sequence.count(aa)
    total = float(len(sequence)) or 1.0
    return counts / total


def compute_physicochemical_props(sequence: str) -> np.ndarray:
    """Compute simple physicochemical summary statistics."""
    length = float(len(sequence))
    hydrophobic = sum(aa in HYDROPHOBIC for aa in sequence)
    polar = sum(aa in POLAR for aa in sequence)
    positive = sum(aa in POSITIVELY_CHARGED for aa in sequence)
    negative = sum(aa in NEGATIVELY_CHARGED for aa in sequence)
    aromatic = sum(aa in "FWY" for aa in sequence)

    return np.array(
        [
            length,
            _safe_divide(hydrophobic, length),
            _safe_divide(polar, length),
            _safe_divide(positive, length),
            _safe_divide(negative, length),
            _safe_divide(aromatic, length),
        ],
        dtype=np.float32,
    )


def compute_features(sequence: str) -> np.ndarray:
    """Generate the final feature vector for a single sequence."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Sequences must contain at least one amino-acid residue.")

    composition = compute_composition(sequence)
    physchem = compute_physicochemical_props(sequence)
    return np.concatenate([composition, physchem])


def build_feature_matrix(sequences: Iterable[str]) -> np.ndarray:
    """Transform an iterable of sequences into a 2D numpy array."""
    features: List[np.ndarray] = []
    for seq in sequences:
        features.append(compute_features(seq))
    if not features:
        return np.zeros((0, len(AMINO_ACIDS) + 6), dtype=np.float32)
    return np.vstack(features)

