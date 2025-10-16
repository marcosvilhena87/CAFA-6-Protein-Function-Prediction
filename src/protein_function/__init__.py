"""Utility package for baseline protein function prediction."""

from .model import ProteinFunctionModel, ModelBundle
from .features import build_feature_matrix
from .data import load_training_data, read_fasta_sequences

__all__ = [
    "ProteinFunctionModel",
    "ModelBundle",
    "build_feature_matrix",
    "load_training_data",
    "read_fasta_sequences",
]
