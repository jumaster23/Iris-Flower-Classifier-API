# 🌸 Iris Flower Classifier API

A production-style REST API that trains and serves Machine Learning models on the classic UCI Iris dataset, built with **FastAPI**, **scikit-learn**, and **uv**.

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [Authentication](#authentication)
- [Models](#models)
- [Running Tests](#running-tests)
- [Docker](#docker)

---

## Overview

This project exposes three trained ML classifiers via a REST API. Given the four measurements of an iris flower (sepal length, sepal width, petal length, petal width), the API predicts the species: **setosa**, **versicolor**, or **virginica**.

---

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [scikit-learn](https://scikit-learn.org/) — ML models
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Pydantic v2](https://docs.pydantic.dev/) — Data validation
- [pytest](https://pytest.org/) — Testing
- [Docker](https://www.docker.com/) — Containerization

---

## Project Structure

```
iris-classifier/
├── main.py              # FastAPI app and endpoints
├── train.py             # Training script
├── inference.py         # Model loading and prediction logic
├── auth.py              # API Key authentication
├── settings.py          # Configuration from .env
├── Dockerfile
├── pyproject.toml
├── .env.example
├── tests/
│   └── test_api.py      # 11 tests
└── saved_models/        # Generated after running train.py
    ├── random_forest.joblib
    ├── logistic_regression.joblib
    ├── svm.joblib
    ├── scaler.joblib
    └── metadata.json
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/iris-classifier.git
cd iris-classifier
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```env
MODEL_FILENAME=random_forest.joblib
SCALER_FILENAME=scaler.joblib
HOST=127.0.0.1
PORT=8001
API_KEY=your-secret-key
```

### 4. Train the models

```bash
uv run train.py
```

### 5. Start the server

```bash
uv run main.py
```

Server runs at `http://127.0.0.1:8001`  
Interactive docs at `http://127.0.0.1:8001/docs`

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | ❌ | Health check |
| `GET` | `/info` | ❌ | Model metadata and accuracy |
| `GET` | `/models` | ❌ | List available models |
| `POST` | `/predict` | ✅ | Single prediction |
| `POST` | `/predict/batch` | ✅ | Batch predictions (max 100) |
| `POST` | `/predict/{model_name}` | ✅ | Predict with a specific model |

### Single prediction

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "predicted_class": "setosa",
  "class_index": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "input": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

### Predict with a specific model

```bash
curl -X POST http://127.0.0.1:8001/predict/svm \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "sepal_length": 6.3,
    "sepal_width": 2.5,
    "petal_length": 4.9,
    "petal_width": 1.5
  }'
```

---

## Authentication

Protected endpoints require an `X-API-Key` header.

```
X-API-Key: your-secret-key
```

Requests without a valid key return `401 Unauthorized`.

To test via Swagger UI (`/docs`), click the **Authorize** 🔓 button and enter your key.

---

## Models

Three models are trained and available:

| Model | Test Accuracy | CV Mean (5-fold) |
|-------|--------------|-----------------|
| Random Forest | 93.3% | 96.7% |
| Logistic Regression | 93.3% | 96.0% |
| SVM (RBF kernel) | 96.7% | 96.7% |

Switch the active model via `.env`:

```env
MODEL_FILENAME=svm.joblib
```

Or call any model directly via the URL:

```
POST /predict/svm
POST /predict/random_forest
POST /predict/logistic_regression
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

```
tests/test_api.py::test_health_check               PASSED
tests/test_api.py::test_model_info                 PASSED
tests/test_api.py::test_list_models                PASSED
tests/test_api.py::test_predict_setosa             PASSED
tests/test_api.py::test_predict_virginica          PASSED
tests/test_api.py::test_predict_invalid_input      PASSED
tests/test_api.py::test_predict_missing_field      PASSED
tests/test_api.py::test_predict_batch              PASSED
tests/test_api.py::test_predict_batch_empty        PASSED
tests/test_api.py::test_predict_with_svm           PASSED
tests/test_api.py::test_predict_with_invalid_model PASSED

11 passed in 1.80s
```

---

## Docker

```bash
# Build
docker build -t iris-classifier .

# Run
docker run -p 8001:8001 iris-classifier
```

---

## Dataset

The [UCI Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) — 150 samples, 3 classes, 4 features:

| Feature | Description |
|---------|-------------|
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |