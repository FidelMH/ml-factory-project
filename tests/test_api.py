import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "api"))

from main import app, state

client = TestClient(app)


def make_mock_model(prediction=0, probas=None):
    """Crée un faux modèle MLflow."""
    if probas is None:
        probas = np.array([[0.9, 0.05, 0.05]])
    model = MagicMock()
    model.predict.return_value = [prediction]
    model._model_impl.predict_proba.return_value = probas
    return model


def reset_state():
    state["model"] = None
    state["version"] = None


# ── 1. Health check ──────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


# ── 2. /model_infos ──────────────────────────────────────────────────────────

@patch("mlflow.pyfunc.load_model")
@patch("mlflow.tracking.MlflowClient.get_model_version_by_alias")
def test_model_infos_success(mock_alias, mock_load):
    mock_alias.return_value = MagicMock(version="1")
    mock_load.return_value = make_mock_model()
    reset_state()

    response = client.get("/model_infos")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_infos"]["version"] == "1"


@patch("mlflow.tracking.MlflowClient.get_model_version_by_alias")
def test_model_infos_mlflow_unavailable(mock_alias):
    mock_alias.side_effect = Exception("MLflow unreachable")
    reset_state()

    response = client.get("/model_infos")
    assert response.status_code == 404


# ── 3. /predict ──────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


@patch("mlflow.pyfunc.load_model")
@patch("mlflow.tracking.MlflowClient.get_model_version_by_alias")
def test_predict_success(mock_alias, mock_load):
    mock_alias.return_value = MagicMock(version="1")
    mock_load.return_value = make_mock_model(
        prediction=0,
        probas=np.array([[0.9, 0.05, 0.05]])
    )
    reset_state()

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert len(data["probabilities"]) == 3
    assert data["version"] == "1"


def test_predict_invalid_type():
    payload = {**VALID_PAYLOAD, "sepal_length": "not_a_float"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "petal_width"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
