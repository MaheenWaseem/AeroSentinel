from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import (
    OUTPUTS_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.data.load_data import (
    load_test_data,
)
from src.inference.rul_inference_service import (
    RULInferenceService,
)
from src.modeling.prepare_sequence_data import (
    WINDOW_LENGTH,
)


FROZEN_PREDICTIONS_PATH = (
    OUTPUTS_DIR
    / "fd001_official_test_predictions_unscored.csv"
)


def expect_value_error(
    check_name: str,
    function: Callable[[], object],
    expected_message_text: str,
) -> None:

    try:
        function()

    except ValueError as error:

        error_message = str(
            error
        )

        if (
            expected_message_text
            not in error_message
        ):
            raise AssertionError(
                f"{check_name} raised ValueError, "
                "but the message was unexpected: "
                f"{error_message}"
            ) from error

        print(
            f"{check_name}: PASSED"
        )

        print(
            f"  {error_message}"
        )

        return

    raise AssertionError(
        f"{check_name} did not raise ValueError."
    )


def validate_result_structure(
    result: dict[
        str,
        int | float,
    ],
) -> None:

    expected_keys = {
        "engine_id",
        "last_observed_cycle",
        "predicted_rul",
    }

    if set(
        result.keys()
    ) != expected_keys:
        raise AssertionError(
            "Prediction result has unexpected keys."
        )

    if not isinstance(
        result["engine_id"],
        int,
    ):
        raise AssertionError(
            "engine_id must be an integer."
        )

    if not isinstance(
        result["last_observed_cycle"],
        int,
    ):
        raise AssertionError(
            "last_observed_cycle must be an int."
        )

    if not isinstance(
        result["predicted_rul"],
        float,
    ):
        raise AssertionError(
            "predicted_rul must be a float."
        )

    if not np.isfinite(
        result["predicted_rul"]
    ):
        raise AssertionError(
            "predicted_rul must be finite."
        )


def main() -> None:

    print(
        "\n======================================"
    )

    print(
        "FD001 PRODUCTION INFERENCE VALIDATION"
    )

    print(
        "======================================"
    )

    service = RULInferenceService()

    test_df = load_test_data()

    if not FROZEN_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Frozen official-test prediction file "
            "was not found: "
            f"{FROZEN_PREDICTIONS_PATH}"
        )

    frozen_predictions_df = pd.read_csv(
        FROZEN_PREDICTIONS_PATH
    )

    required_prediction_columns = {
        "engine_id",
        "last_observed_cycle",
        "predicted_rul",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(
            frozen_predictions_df.columns
        )
    )

    if missing_prediction_columns:
        raise ValueError(
            "Frozen prediction file is missing "
            "required columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    engine_lengths = (
        test_df
        .groupby(
            "engine_id"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    selected_engine_id = int(
        engine_lengths.index[0]
    )

    selected_engine_length = int(
        engine_lengths.iloc[0]
    )

    if (
        selected_engine_length
        <= WINDOW_LENGTH
    ):
        raise ValueError(
            "A test engine with more than the "
            "minimum history length was expected."
        )

    engine_history = (
        test_df.loc[
            test_df[
                "engine_id"
            ]
            == selected_engine_id
        ]
        .copy()
    )

    engine_history = (
        engine_history
        .sort_values(
            "cycle"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nSelected validation engine:"
    )

    print(
        selected_engine_id
    )

    print(
        "\nAvailable history rows:"
    )

    print(
        len(
            engine_history
        )
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 1 — HISTORY > 30 CYCLES"
    )

    print(
        "--------------------------------------"
    )

    full_history_result = (
        service.predict(
            engine_history
        )
    )

    validate_result_structure(
        full_history_result
    )

    print(
        "PASSED"
    )

    print(
        full_history_result
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 2 — EXACTLY 30 CYCLES"
    )

    print(
        "--------------------------------------"
    )

    exact_window_history = (
        engine_history
        .tail(
            WINDOW_LENGTH
        )
        .copy()
    )

    exact_window_result = (
        service.predict(
            exact_window_history
        )
    )

    validate_result_structure(
        exact_window_result
    )

    if not np.isclose(
        full_history_result[
            "predicted_rul"
        ],
        exact_window_result[
            "predicted_rul"
        ],
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError(
            "Full-history and final-30 predictions "
            "do not match."
        )

    print(
        "PASSED"
    )

    print(
        exact_window_result
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 3 — ONLY FINAL 30 CYCLES USED"
    )

    print(
        "--------------------------------------"
    )

    modified_early_history = (
        engine_history.copy()
    )

    early_row_count = (
        len(
            modified_early_history
        )
        - WINDOW_LENGTH
    )

    modified_early_history.loc[
        :early_row_count - 1,
        USEFUL_SENSOR_COLUMNS,
    ] = (
        modified_early_history.loc[
            :early_row_count - 1,
            USEFUL_SENSOR_COLUMNS,
        ]
        + 999999.0
    )

    modified_history_result = (
        service.predict(
            modified_early_history
        )
    )

    if not np.isclose(
        full_history_result[
            "predicted_rul"
        ],
        modified_history_result[
            "predicted_rul"
        ],
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError(
            "Changing observations before the final "
            "30 cycles changed the prediction."
        )

    print(
        "PASSED"
    )

    print(
        "Earlier observations were changed, "
        "but the prediction remained unchanged."
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 4 — OUTPUT STRUCTURE"
    )

    print(
        "--------------------------------------"
    )

    validate_result_structure(
        full_history_result
    )

    print(
        "PASSED"
    )

    print(
        "engine_id: int"
    )

    print(
        "last_observed_cycle: int"
    )

    print(
        "predicted_rul: float"
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 5 — DETERMINISTIC INFERENCE"
    )

    print(
        "--------------------------------------"
    )

    repeated_result_1 = (
        service.predict(
            engine_history
        )
    )

    repeated_result_2 = (
        service.predict(
            engine_history
        )
    )

    if (
        repeated_result_1[
            "predicted_rul"
        ]
        != repeated_result_2[
            "predicted_rul"
        ]
    ):
        raise AssertionError(
            "Repeated identical inference produced "
            "different predictions."
        )

    print(
        "PASSED"
    )

    print(
        "Prediction 1:"
    )

    print(
        repeated_result_1[
            "predicted_rul"
        ]
    )

    print(
        "Prediction 2:"
    )

    print(
        repeated_result_2[
            "predicted_rul"
        ]
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 6 — INSUFFICIENT HISTORY"
    )

    print(
        "--------------------------------------"
    )

    insufficient_history = (
        exact_window_history
        .iloc[
            :-1
        ]
        .copy()
    )

    expect_value_error(
        "Insufficient-history check",
        lambda: service.predict(
            insufficient_history
        ),
        "at least 30 observations",
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 7 — MISSING SENSOR"
    )

    print(
        "--------------------------------------"
    )

    missing_sensor_history = (
        exact_window_history
        .drop(
            columns=[
                USEFUL_SENSOR_COLUMNS[0]
            ]
        )
    )

    expect_value_error(
        "Missing-sensor check",
        lambda: service.predict(
            missing_sensor_history
        ),
        "missing required columns",
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 8 — NaN INPUT"
    )

    print(
        "--------------------------------------"
    )

    nan_history = (
        exact_window_history.copy()
    )

    nan_history.loc[
        nan_history.index[0],
        USEFUL_SENSOR_COLUMNS[0],
    ] = np.nan

    expect_value_error(
        "NaN check",
        lambda: service.predict(
            nan_history
        ),
        "contains missing values",
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 9 — DUPLICATE CYCLE"
    )

    print(
        "--------------------------------------"
    )

    duplicate_cycle_history = pd.concat(
        [
            exact_window_history,
            exact_window_history.tail(
                1
            ),
        ],
        ignore_index=True,
    )

    expect_value_error(
        "Duplicate-cycle check",
        lambda: service.predict(
            duplicate_cycle_history
        ),
        "duplicate cycle values",
    )

    print(
        "\n--------------------------------------"
    )

    print(
        "CHECK 10 — FROZEN PREDICTION"
    )

    print(
        "--------------------------------------"
    )

    frozen_engine_rows = (
        frozen_predictions_df.loc[
            frozen_predictions_df[
                "engine_id"
            ]
            == selected_engine_id
        ]
    )

    if len(
        frozen_engine_rows
    ) != 1:
        raise ValueError(
            "Expected exactly one frozen prediction "
            "for the selected engine."
        )

    frozen_row = (
        frozen_engine_rows
        .iloc[0]
    )

    frozen_last_cycle = int(
        frozen_row[
            "last_observed_cycle"
        ]
    )

    frozen_prediction = float(
        frozen_row[
            "predicted_rul"
        ]
    )

    current_last_cycle = (
        full_history_result[
            "last_observed_cycle"
        ]
    )

    current_prediction = (
        full_history_result[
            "predicted_rul"
        ]
    )

    if (
        frozen_last_cycle
        != current_last_cycle
    ):
        raise AssertionError(
            "Production inference last cycle does "
            "not match the frozen prediction record."
        )

    prediction_difference = abs(
        current_prediction
        - frozen_prediction
    )

    if not np.isclose(
        current_prediction,
        frozen_prediction,
        rtol=1e-6,
        atol=1e-4,
    ):
        raise AssertionError(
            "Production inference does not reproduce "
            "the frozen official-test prediction. "
            f"Difference: {prediction_difference}"
        )

    print(
        "PASSED"
    )

    print(
        "Frozen prediction:"
    )

    print(
        frozen_prediction
    )

    print(
        "Production-service prediction:"
    )

    print(
        current_prediction
    )

    print(
        "Absolute difference:"
    )

    print(
        prediction_difference
    )

    print(
        "\n======================================"
    )

    print(
        "ALL INFERENCE CHECKS PASSED"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main() 