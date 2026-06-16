# tests/test_api.py

import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app, raise_server_exceptions=True)

WINDOW_SIZE  = 144
NUM_FEATURES = 19

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def dummy_window(rows=WINDOW_SIZE, cols=NUM_FEATURES):
    """Genera una ventana de ceros con las dimensiones correctas."""
    return np.zeros((rows, cols)).tolist()


# -------------------------------------------------------
# /health
# -------------------------------------------------------
def test_health_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -------------------------------------------------------
# /info
# -------------------------------------------------------
def test_info_returns_200(client):
    response = client.get("/info")
    assert response.status_code == 200

def test_info_fields(client):
    response = client.get("/info")
    data = response.json()
    assert data["model"]        == "LSTM_p3_w144_h64_l1"
    assert data["window_size"]  == WINDOW_SIZE
    assert data["num_features"] == NUM_FEATURES
    assert data["horizon"]      == 1
    assert data["units"]        == "°C"
    assert "test_mae"  in data
    assert "test_rmse" in data

def test_info_metrics_are_positive(client):
    response = client.get("/info")
    data = response.json()
    assert data["test_mae"]  > 0
    assert data["test_rmse"] > 0


# -------------------------------------------------------
# /predict — casos válidos
# -------------------------------------------------------
def test_predict_returns_200(client):
    response = client.post("/predict", json={"window": dummy_window()})
    assert response.status_code == 200

def test_predict_response_fields(client):
    response = client.post("/predict", json={"window": dummy_window()})
    data = response.json()
    assert "predicted_temperature_celsius" in data
    assert "model"                         in data
    assert "horizon_minutes"               in data

def test_predict_temperature_in_reasonable_range(client):
    response = client.post("/predict", json={"window": dummy_window()})
    temp = response.json()["predicted_temperature_celsius"]
    assert -50.0 <= temp <= 60.0

def test_predict_horizon_is_10_minutes(client):
    response = client.post("/predict", json={"window": dummy_window()})
    assert response.json()["horizon_minutes"] == 10


# -------------------------------------------------------
# /predict — casos inválidos
# -------------------------------------------------------
def test_predict_wrong_window_size_returns_422(client):
    bad_window = dummy_window(rows=10)  # solo 10 pasos en lugar de 144
    response = client.post("/predict", json={"window": bad_window})
    assert response.status_code == 422

def test_predict_wrong_num_features_returns_422(client):
    bad_window = dummy_window(cols=5)  # solo 5 features en lugar de 19
    response = client.post("/predict", json={"window": bad_window})
    assert response.status_code == 422

def test_predict_empty_window_returns_422(client):
    response = client.post("/predict", json={"window": []})
    assert response.status_code == 422
