import numpy as np
import pandas as pd
import pytest

from src.config import USEFUL_SENSOR_COLUMNS
from src.modeling.prepare_sequence_data import WINDOW_LENGTH


@pytest.fixture
def valid_engine_history():
    """Create deterministic, valid engine history."""

    cycles = np.arange(1, WINDOW_LENGTH + 1)

    data = {
        "engine_id": np.ones(WINDOW_LENGTH, dtype=int),
        "cycle": cycles,
    }

    for index, sensor in enumerate(USEFUL_SENSOR_COLUMNS):
        data[sensor] = (
            100.0 + index + cycles * 0.1
        )

    return pd.DataFrame(data)