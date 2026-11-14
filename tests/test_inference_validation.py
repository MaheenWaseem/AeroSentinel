import numpy as np
import pandas as pd
import pytest

from src.config import USEFUL_SENSOR_COLUMNS
from src.inference.rul_inference_service import RULInferenceService
from src.modeling.prepare_sequence_data import WINDOW_LENGTH


@pytest.fixture
def validation_service():
    """
    Create a service instance without loading ML artifacts.

    Safe for these tests because _validate_engine_history()
    does not use initialized service attributes.
    """
    return object.__new__(RULInferenceService)


def test_valid_history_is_accepted(
    validation_service,
    valid_engine_history,
):
    result = validation_service._validate_engine_history(
        valid_engine_history
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == WINDOW_LENGTH
    assert result["engine_id"].nunique() == 1
    assert len(USEFUL_SENSOR_COLUMNS) == 14


def test_history_is_sorted_by_cycle(
    validation_service,
    valid_engine_history,
):
    shuffled_history = valid_engine_history.sample(
        frac=1,
        random_state=42,
    )

    result = validation_service._validate_engine_history(
        shuffled_history
    )

    assert result["cycle"].is_monotonic_increasing
    assert result["cycle"].iloc[0] == 1
    assert result["cycle"].iloc[-1] == WINDOW_LENGTH


def test_missing_sensor_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.drop(
        columns=[USEFUL_SENSOR_COLUMNS[0]]
    )

    with pytest.raises(ValueError, match="missing required"):
        validation_service._validate_engine_history(history)


def test_multiple_engines_are_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()
    history.loc[0, "engine_id"] = 2

    with pytest.raises(ValueError, match="exactly one engine"):
        validation_service._validate_engine_history(history)


def test_short_history_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.iloc[
        : WINDOW_LENGTH - 1
    ].copy()

    with pytest.raises(ValueError, match="at least"):
        validation_service._validate_engine_history(history)


def test_duplicate_cycles_are_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()
    history.loc[1, "cycle"] = 1

    with pytest.raises(ValueError, match="duplicate"):
        validation_service._validate_engine_history(history)


def test_nonpositive_cycle_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()
    history.loc[0, "cycle"] = 0

    with pytest.raises(ValueError, match="positive"):
        validation_service._validate_engine_history(history)


def test_noninteger_cycle_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()

    history["cycle"] = history["cycle"].astype(float)

    history.loc[0, "cycle"] = 1.5

    with pytest.raises(ValueError, match="integers"):
        validation_service._validate_engine_history(history)

def test_missing_sensor_value_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()
    history.loc[0, USEFUL_SENSOR_COLUMNS[0]] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validation_service._validate_engine_history(history)


def test_nonfinite_sensor_value_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.copy()
    history.loc[0, USEFUL_SENSOR_COLUMNS[0]] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        validation_service._validate_engine_history(history)


def test_empty_history_is_rejected(
    validation_service,
    valid_engine_history,
):
    history = valid_engine_history.iloc[0:0].copy()

    with pytest.raises(ValueError, match="cannot be empty"):
        validation_service._validate_engine_history(history)


def test_dataframe_is_required(
    validation_service,
):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validation_service._validate_engine_history(
            {"engine_id": [1]}
        )


def test_original_history_is_not_modified(
    validation_service,
    valid_engine_history,
):
    original = valid_engine_history.copy(deep=True)

    validation_service._validate_engine_history(
        valid_engine_history
    )

    pd.testing.assert_frame_equal(
        valid_engine_history,
        original,
    )