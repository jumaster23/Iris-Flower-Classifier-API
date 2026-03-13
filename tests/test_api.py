"""
tests/test_api.py — Tests for the Iris Classifier API.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

SETOSA_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

VIRGINICA_INPUT = {
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 6.0,
    "petal_width": 2.5,
}


@pytest.fixture(scope="session")
def client():
    """Cliente que arranca el lifespan completo (carga el modelo)."""
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True


# --------------------------------------------------------------------------- #
# Model info
# --------------------------------------------------------------------------- #

def test_model_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "classes" in data


def test_list_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "random_forest" in data["available_models"]
    assert "svm" in data["available_models"]


# --------------------------------------------------------------------------- #
# Single prediction
# --------------------------------------------------------------------------- #

def test_predict_setosa(client):
    response = client.post("/predict", json=SETOSA_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "setosa"
    assert data["class_index"] == 0
    assert set(data["probabilities"].keys()) == {"setosa", "versicolor", "virginica"}


def test_predict_virginica(client):
    response = client.post("/predict", json=VIRGINICA_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "virginica"


def test_predict_invalid_input(client):
    response = client.post("/predict", json={"sepal_length": -1.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert response.status_code == 422


def test_predict_missing_field(client):
    response = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Batch prediction
# --------------------------------------------------------------------------- #

def test_predict_batch(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA_INPUT, VIRGINICA_INPUT]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["predictions"][0]["predicted_class"] == "setosa"
    assert data["predictions"][1]["predicted_class"] == "virginica"


def test_predict_batch_empty(client):
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Predict with specific model
# --------------------------------------------------------------------------- #

def test_predict_with_svm(client):
    response = client.post("/predict/svm", json=SETOSA_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "setosa"


def test_predict_with_invalid_model(client):
    response = client.post("/predict/nonexistent_model", json=SETOSA_INPUT)
    assert response.status_code == 404