import math

import numpy as np
import pandas as pd
import pytest
import torch

import src.inference.rul_inference_service as inference_module

from src.config import USEFUL_SENSOR_COLUMNS
from src.inference.rul_inference_service import RULInferenceService
from src.modeling.prepare_sequence_data import WINDOW_LENGTH


# --------------------------------------------------
# Shared inference service
# --------------------------------------------------

@pytest.fixture(scope="module")
def inference_service():
 
    return RULInferenceService()


# --------------------------------------------------
# 1. Artifact and model initialization
# --------------------------------------------------

def test_inference_service_initializes(inference_service):

    assert inference_service.sensor_scaler is not None
    assert inference_service.cycle_scaler is not None
    assert inference_service.checkpoint is not None
    assert inference_service.model is not None

    assert inference_service.model.training is False

    assert inference_service.device.type == "cpu"


# --------------------------------------------------
# 2. Artifact compatibility
# --------------------------------------------------

def test_loaded_artifacts_match_locked_configuration(
    inference_service,
):

    assert (
        inference_service.sensor_scaler.n_features_in_
        == len(USEFUL_SENSOR_COLUMNS)
    )

    assert (
        inference_service.sensor_scaler
        .feature_names_in_.tolist()
        == USEFUL_SENSOR_COLUMNS
    )

    assert (
        inference_service.cycle_scaler.n_features_in_
        == 1
    )

    assert (
        inference_service.checkpoint["window_length"]
        == WINDOW_LENGTH
    )

    assert (
        inference_service.checkpoint["sensor_columns"]
        == USEFUL_SENSOR_COLUMNS
    )

    assert (
        inference_service.checkpoint["training_engine_count"]
        == 100
    )


# --------------------------------------------------
# 3. Successful real inference
# --------------------------------------------------

def test_valid_history_produces_prediction(
    inference_service,
    valid_engine_history,
):

    result = inference_service.predict(
        valid_engine_history
    )

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "engine_id",
        "last_observed_cycle",
        "predicted_rul",
    }

    assert result["engine_id"] == 1

    assert (
        result["last_observed_cycle"]
        == WINDOW_LENGTH
    )

    assert isinstance(
        result["predicted_rul"],
        float,
    )

    assert math.isfinite(
        result["predicted_rul"]
    )


# --------------------------------------------------
# 4. Longer engine histories
# --------------------------------------------------

def test_long_history_uses_latest_window(
    inference_service,
    valid_engine_history,
):

    extra = valid_engine_history.iloc[:5].copy()

    extra["cycle"] = np.arange(
        WINDOW_LENGTH + 1,
        WINDOW_LENGTH + 6,
    )

    longer_history = pd.concat(
        [valid_engine_history, extra],
        ignore_index=True,
    )

    full_result = inference_service.predict(
        longer_history
    )

    latest_window = longer_history.tail(
        WINDOW_LENGTH
    ).copy()

    window_result = inference_service.predict(
        latest_window
    )

    assert full_result["last_observed_cycle"] == (
        WINDOW_LENGTH + 5
    )

    assert full_result["engine_id"] == 1

    assert full_result["predicted_rul"] == pytest.approx(
        window_result["predicted_rul"],
        abs=1e-5,
    )


# --------------------------------------------------
# 5. Ordering does not change prediction
# --------------------------------------------------

def test_prediction_is_independent_of_row_order(
    inference_service,
    valid_engine_history,
):

    shuffled_history = valid_engine_history.sample(
        frac=1,
        random_state=42,
    )

    original_result = inference_service.predict(
        valid_engine_history
    )

    shuffled_result = inference_service.predict(
        shuffled_history
    )

    assert shuffled_result["last_observed_cycle"] == (
        original_result["last_observed_cycle"]
    )

    assert shuffled_result["predicted_rul"] == pytest.approx(
        original_result["predicted_rul"],
        abs=1e-5,
    )


# --------------------------------------------------
# 6. Missing artifact handling
# --------------------------------------------------

def test_missing_artifact_raises_clear_error(
    monkeypatch,
    tmp_path,
):

    missing_model = tmp_path / "missing_model.pt"

    monkeypatch.setattr(
        inference_module,
        "FINAL_MODEL_PATH",
        missing_model,
    )

    service = object.__new__(RULInferenceService)

    with pytest.raises(
        FileNotFoundError,
        match="Required final inference artifacts",
    ):
        service._validate_artifact_files()


# --------------------------------------------------
# 7. Prediction tensor preparation
# --------------------------------------------------

def test_model_input_tensor_shapes(
    inference_service,
    valid_engine_history,
):

    validated = inference_service._validate_engine_history(
        valid_engine_history
    )

    (
        sensor_tensor,
        cycle_tensor,
        engine_id,
        last_observed_cycle,
    ) = inference_service._prepare_model_inputs(
        validated
    )

    assert sensor_tensor.shape == (
        1,
        WINDOW_LENGTH,
        len(USEFUL_SENSOR_COLUMNS),
    )

    assert cycle_tensor.shape == (1, 1)

    assert sensor_tensor.dtype == torch.float32
    assert cycle_tensor.dtype == torch.float32

    assert torch.isfinite(sensor_tensor).all().item()
    assert torch.isfinite(cycle_tensor).all().item()

    assert engine_id == 1
    assert last_observed_cycle == WINDOW_LENGTH