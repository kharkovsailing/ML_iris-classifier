# ML_iris-classifier
# Iris Classifier – Dockerized Deep Learning Pipeline

This repository contains a complete pipeline for training and running inference on the Iris flower classification dataset using PyTorch. It is fully Dockerized and includes unit tests and exception handling.

## Repository Structure

```
.
├── data/                    # CSV files for training and inference
├── training/                # Training container code
├── inference/               # Inference container code
├── .gitignore
└── README.md
```

## Usage

### 1. Train the Model

```bash
cd training
docker build -t iris-train .
docker run --rm -v "$PWD/../":/app iris-train
```

### 2. Run Inference

```bash
cd ../inference
docker build -t iris-infer .
docker run --rm -v "$PWD/../":/app iris-infer
```

### 3. Run Tests

```bash
docker run --rm -v "$PWD/../":/app -w /app/training iris-train pytest test_train.py
docker run --rm -v "$PWD/../":/app -w /app/inference iris-infer pytest test_infer.py
```
