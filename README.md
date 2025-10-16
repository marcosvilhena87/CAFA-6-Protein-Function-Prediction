# CAFA 6 Protein Function Prediction Baseline

This repository provides a light-weight, fully documented baseline for the [CAFA 6 protein function prediction challenge](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction). The goal is to map amino acid sequences to Gene Ontology (GO) terms. The supplied code focuses on clarity and reproducibility so that it can serve as a starting point for further experimentation.

## Features

- Pure Python / scikit-learn implementation with no specialized hardware requirements.
- Simple amino-acid composition features that work for sequences of arbitrary length.
- End-to-end command line tools for training and inference.
- Predicts calibrated probabilities that can be converted directly into the submission TSV format.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── protein_function/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
└── tests/            # optional, add your own unit tests here
```

## Installation

Create a virtual environment (recommended) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input data format

The baseline expects two kinds of input files:

1. **Training FASTA file** – Contains amino acid sequences whose functions are known.
2. **Annotation CSV file** – Contains two columns: `sequence_id` and `go_term`. Each row associates a sequence with a GO term. Sequences may appear multiple times, once per GO term.

At inference time you only need a FASTA file with the sequences to annotate.

## Training a model

```bash
python -m protein_function.train \
    --fasta data/train_sequences.fasta \
    --annotations data/train_annotations.csv \
    --output models/baseline.joblib
```

The command:

1. Parses the FASTA file and annotations.
2. Extracts amino-acid composition features.
3. Trains a multi-label logistic regression model.
4. Saves the fitted model (classifier + label encoder) into a `.joblib` bundle.

## Generating predictions

```bash
python -m protein_function.predict \
    --model models/baseline.joblib \
    --fasta data/test_sequences.fasta \
    --output predictions.tsv \
    --top-k 25
```

The prediction script outputs a tab-separated file that follows the competition submission format:

```
sequence_id\tGO:0000001\t0.123
sequence_id\tGO:0000002\t0.056
...
```

You can control the number of GO terms emitted per sequence via `--top-k`, or specify a probability threshold with `--min-proba`.

## Extending the baseline

- Swap the simple feature extractor with a deep representation (e.g., ESM embeddings).
- Replace the linear classifier with gradient boosted trees or neural networks.
- Add ontology-aware post-processing, such as propagating scores to parent GO terms.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
