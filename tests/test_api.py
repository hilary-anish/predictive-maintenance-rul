"""
Test FastAPI endpoints.

Why test the API? The API is the public interface to your model.
If it returns wrong data, wrong status codes, or crashes on
edge cases, nothing downstream works.

Uses FastAPI's TestClient which simulates HTTP requests
without actually starting a server — fast and reliable.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """
    Create a test client with a mocked model.

    Why mock? Tests should be fast and deterministic.
    Loading a real PyTorch model takes seconds and requires
    GPU/data files. Mocking lets us test API logic in isolation.
    """
    # Mock the model to return a fixed prediction
    mock_model = MagicMock()
    mock_model.eval = MagicMock()

    import torch
    mock_model.return_value = torch.tensor([65.0])
    mock_model.parameters = MagicMock(
        return_value=iter([torch.tensor([1.0])])
    )

    # Mock conformal predictor
    mock_conformal = MagicMock()
    mock_conformal.predict = MagicMock(return_value={
        "lower": np.array([42.0]),
        "upper": np.array([88.0]),
        "point": np.array([65.0]),
        "width": np.array([46.0]),
        "q_hat": 23.0,
    })

    # Patch the startup to inject mocks
    from pdm.serving.api import STATE, app
    STATE["model"] = mock_model
    STATE["model_info"] = {
        "name": "rul-lstm",
        "version": "test",
        "stage": "test",
    }
    STATE["conformal"] = mock_conformal

    return TestClient(app)


def test_health(client):
    """Health endpoint should return 200 with model info."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "rul-lstm"


def test_predict_valid_input(client):
    """POST /predict with valid input should return RUL + intervals."""
    sequence = [[0.5] * 14] * 30  # 30 timesteps, 14 features
    resp = client.post("/predict", json={"sequence": sequence})
    assert resp.status_code == 200
    data = resp.json()
    assert "rul" in data
    assert "lower" in data
    assert "upper" in data
    assert "unit_status" in data
    assert data["lower"] <= data["rul"] <= data["upper"]


def test_predict_wrong_shape(client):
    """POST /predict with wrong shape should return 422."""
    sequence = [[0.5] * 10] * 30  # 10 features instead of 14
    resp = client.post("/predict", json={"sequence": sequence})
    assert resp.status_code == 422
