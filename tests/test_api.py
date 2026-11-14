import math

import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas.predict import PredictionResponse
from src.modeling.prepare_sequence_data import WINDOW_LENGTH


# --------------------------------------------------
# Shared API client
# --------------------------------------------------

@pytest.fixture(scope="module")
def client():
    

    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------
# Request payload fixture
# --------------------------------------------------

@pytest.fixture
def valid_prediction_payload(valid_engine_history):
   

    history = valid_engine_history.drop(
        columns=["engine_id"]
    )

    return {
        "engine_id": 1,
        "history": history.to_dict(
            orient="records"
        ),
    }


# --------------------------------------------------
# 1. Health endpoint
# --------------------------------------------------

def test_health_returns_ok(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


# --------------------------------------------------
# 2. Application startup
# --------------------------------------------------

def test_inference_service_available_on_startup(client):

    service = client.app.state.rul_service

    assert service is not None

    assert service.model is not None

    assert service.sensor_scaler is not None

    assert service.cycle_scaler is not None

    assert service.model.training is False


# --------------------------------------------------
# 3. Successful prediction
# --------------------------------------------------

def test_predict_returns_success(
    client,
    valid_prediction_payload,
):

    response = client.post(
        "/predict",
        json=valid_prediction_payload,
    )

    assert response.status_code == 200

    result = response.json()

    validated_response = PredictionResponse.model_validate(
        result
    )

    assert set(result.keys()) == {
        "engine_id",
        "last_observed_cycle",
        "predicted_rul",
    }

    assert validated_response.engine_id == 1

    assert (
        validated_response.last_observed_cycle
        == WINDOW_LENGTH
    )

    assert math.isfinite(
        validated_response.predicted_rul
    )


# --------------------------------------------------
# 4. Invalid engine ID → 422
# --------------------------------------------------

def test_predict_rejects_invalid_engine_id(
    client,
    valid_prediction_payload,
):

    payload = valid_prediction_payload.copy()

    payload["engine_id"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# --------------------------------------------------
# 5. Invalid cycle → 422
# --------------------------------------------------

def test_predict_rejects_invalid_cycle(
    client,
    valid_prediction_payload,
):

    payload = {
        **valid_prediction_payload,
        "history": [
            record.copy()
            for record in valid_prediction_payload["history"]
        ],
    }

    payload["history"][0]["cycle"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# --------------------------------------------------
# 6. Missing sensor → 422
# --------------------------------------------------

def test_predict_rejects_missing_sensor(
    client,
    valid_prediction_payload,
):

    payload = {
        **valid_prediction_payload,
        "history": [
            record.copy()
            for record in valid_prediction_payload["history"]
        ],
    }

    del payload["history"][0]["sensor_2"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# --------------------------------------------------
# 7. Empty history → 422
# --------------------------------------------------

def test_predict_rejects_empty_history(
    client,
):

    payload = {
        "engine_id": 1,
        "history": [],
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# --------------------------------------------------
# 8. Too-short history → 400
# --------------------------------------------------

def test_predict_rejects_short_history(
    client,
    valid_prediction_payload,
):

    payload = {
        **valid_prediction_payload,
        "history": valid_prediction_payload["history"][
            : WINDOW_LENGTH - 1
        ],
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    assert "at least" in response.json()["detail"]


# --------------------------------------------------
# 9. Duplicate cycles → 400
# --------------------------------------------------

def test_predict_rejects_duplicate_cycles(
    client,
    valid_prediction_payload,
):

    payload = {
        **valid_prediction_payload,
        "history": [
            record.copy()
            for record in valid_prediction_payload["history"]
        ],
    }

    payload["history"][1]["cycle"] = (
        payload["history"][0]["cycle"]
    )

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    assert "duplicate" in response.json()["detail"]


# --------------------------------------------------
# 10. Unexpected internal failure → 500
# --------------------------------------------------

def test_predict_handles_internal_failure(
    client,
    valid_prediction_payload,
    monkeypatch,
):

    service = client.app.state.rul_service

    def broken_predict(engine_history):
        raise RuntimeError(
            "Simulated internal inference failure"
        )

    monkeypatch.setattr(
        service,
        "predict",
        broken_predict,
    )

    response = client.post(
        "/predict",
        json=valid_prediction_payload,
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Prediction failed due to an "
            "internal server error."
        )
    }