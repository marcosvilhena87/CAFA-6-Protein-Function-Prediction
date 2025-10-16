"""Model wrapper for protein function prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from .features import build_feature_matrix


@dataclass
class ModelBundle:
    """Container storing the classifier and label encoder."""

    classifier: OneVsRestClassifier
    label_binarizer: MultiLabelBinarizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.classifier, "label_binarizer": self.label_binarizer}, path)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        data = joblib.load(path)
        return ModelBundle(
            classifier=data["classifier"], label_binarizer=data["label_binarizer"]
        )


class ProteinFunctionModel:
    """High-level API for training and inference."""

    def __init__(self, penalty: str = "l2", c: float = 1.0, max_iter: int = 1000) -> None:
        base_classifier = LogisticRegression(
            penalty=penalty,
            C=c,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.classifier = OneVsRestClassifier(base_classifier)
        self.label_binarizer = MultiLabelBinarizer()

    def fit(self, sequences: Sequence[str], labels: Sequence[Iterable[str]]) -> ModelBundle:
        X = build_feature_matrix(sequences)
        Y = self.label_binarizer.fit_transform(labels)
        self.classifier.fit(X, Y)
        return ModelBundle(self.classifier, self.label_binarizer)

    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        X = build_feature_matrix(sequences)
        return self.classifier.predict_proba(X)

    def evaluate(
        self, sequences: Sequence[str], labels: Sequence[Iterable[str]], average: str = "micro"
    ) -> float:
        X = build_feature_matrix(sequences)
        Y_true = self.label_binarizer.transform(labels)
        Y_pred = self.classifier.predict(X)
        return f1_score(Y_true, Y_pred, average=average)

    def save(self, path: str | Path) -> None:
        ModelBundle(self.classifier, self.label_binarizer).save(path)

    @classmethod
    def load(cls, path: str | Path) -> "ProteinFunctionModel":
        bundle = ModelBundle.load(path)
        model = cls()
        model.classifier = bundle.classifier
        model.label_binarizer = bundle.label_binarizer
        return model


def train_test_split_examples(
    sequences: Sequence[str], labels: Sequence[Iterable[str]], test_size: float = 0.2, random_state: int = 42
):
    """Utility helper to split sequences and labels for quick experiments."""
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=None
    )
    return X_train, X_val, y_train, y_val

