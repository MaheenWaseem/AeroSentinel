from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.config import (
    OUTPUTS_DIR,
    SAVED_ARTIFACTS_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.data.load_data import (
    load_test_data,
)
from src.modeling.cycle_aware_lstm_model import (
    CycleAwareRULLSTM,
)
from src.modeling.prepare_sequence_data import (
    WINDOW_LENGTH,
    apply_sensor_scaling,
)
from src.modeling.train_cycle_aware_lstm import (
    BATCH_SIZE,
)


LOCK_FILE = (
    OUTPUTS_DIR
    / "lock_development_protocol.json"
)

FINAL_SENSOR_SCALER_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_sequence_sensor_scaler.joblib"
)

FINAL_CYCLE_SCALER_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_cycle_feature_scaler.joblib"
)

FINAL_MODEL_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_cycle_aware_lstm.pt"
)

UNSCORED_PREDICTIONS_PATH = (
    OUTPUTS_DIR
    / "fd001_official_test_predictions_unscored.csv"
)


def load_locked_protocol() -> dict:

    if not LOCK_FILE.exists():
        raise FileNotFoundError(
            f"Lock artifact not found: {LOCK_FILE}"
        )

    with LOCK_FILE.open(
        "r",
        encoding="utf-8",
    ) as json_file:

        protocol = json.load(
            json_file
        )

    if (
        protocol[
            "lock_metadata"
        ][
            "development_phase_status"
        ]
        != "closed"
    ):
        raise ValueError(
            "Development phase is not locked."
        )

    if protocol[
        "lock_metadata"
    ][
        "official_test_ground_truth_used"
    ]:
        raise ValueError(
            "Lock artifact indicates that official "
            "test ground truth was already used."
        )

    return protocol


def validate_final_artifacts() -> None:

    required_paths = [
        FINAL_SENSOR_SCALER_PATH,
        FINAL_CYCLE_SCALER_PATH,
        FINAL_MODEL_PATH,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing final artifacts: "
            f"{missing_paths}"
        )


def build_final_test_inputs(
    test_df: pd.DataFrame,
    sensor_scaler,
    cycle_scaler,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:

    engine_count = int(
        test_df[
            "engine_id"
        ].nunique()
    )

    if engine_count != 100:
        raise ValueError(
            "Official test must contain exactly "
            "100 engines."
        )

    expected_engine_ids = list(
        range(
            1,
            101,
        )
    )

    actual_engine_ids = sorted(
        test_df[
            "engine_id"
        ]
        .unique()
        .tolist()
    )

    if actual_engine_ids != expected_engine_ids:
        raise ValueError(
            "Official test engine IDs are not "
            "exactly 1 through 100."
        )

    engine_lengths = (
        test_df
        .groupby(
            "engine_id"
        )
        .size()
    )

    short_engines = (
        engine_lengths[
            engine_lengths
            < WINDOW_LENGTH
        ]
    )

    if len(short_engines) > 0:
        raise ValueError(
            "Some test engines do not have enough "
            "history for the locked 30-cycle window: "
            f"{short_engines.to_dict()}"
        )

    scaled_test_df = (
        apply_sensor_scaling(
            test_df,
            sensor_scaler,
        )
    )

    sequences = []
    ending_cycles = []
    metadata_rows = []

    for (
        engine_id,
        engine_df,
    ) in scaled_test_df.groupby(
        "engine_id",
        sort=True,
    ):

        engine_df = (
            engine_df
            .sort_values(
                "cycle"
            )
            .reset_index(
                drop=True
            )
        )

        final_window_df = (
            engine_df
            .tail(
                WINDOW_LENGTH
            )
        )

        sequence = (
            final_window_df[
                USEFUL_SENSOR_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if sequence.shape != (
            WINDOW_LENGTH,
            len(
                USEFUL_SENSOR_COLUMNS
            ),
        ):
            raise ValueError(
                "Unexpected final sequence shape "
                f"for engine {engine_id}: "
                f"{sequence.shape}"
            )

        last_observed_cycle = int(
            engine_df[
                "cycle"
            ].iloc[-1]
        )

        sequences.append(
            sequence
        )

        ending_cycles.append(
            last_observed_cycle
        )

        metadata_rows.append(
            {
                "engine_id": int(
                    engine_id
                ),
                "last_observed_cycle": (
                    last_observed_cycle
                ),
            }
        )

    sequences_array = np.stack(
        sequences
    ).astype(
        np.float32
    )

    ending_cycle_array = np.asarray(
        ending_cycles,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    cycle_features = (
        cycle_scaler
        .transform(
            ending_cycle_array
        )
        .astype(
            np.float32
        )
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    if sequences_array.shape != (
        100,
        WINDOW_LENGTH,
        len(
            USEFUL_SENSOR_COLUMNS
        ),
    ):
        raise ValueError(
            "Final official-test sequence array "
            "has an unexpected shape."
        )

    if cycle_features.shape != (
        100,
        1,
    ):
        raise ValueError(
            "Final official-test cycle feature array "
            "has an unexpected shape."
        )

    if len(metadata_df) != 100:
        raise ValueError(
            "Official-test metadata must contain "
            "exactly 100 rows."
        )

    if not np.isfinite(
        sequences_array
    ).all():
        raise ValueError(
            "Official-test sensor sequences contain "
            "non-finite values."
        )

    if not np.isfinite(
        cycle_features
    ).all():
        raise ValueError(
            "Official-test cycle features contain "
            "non-finite values."
        )

    return (
        sequences_array,
        cycle_features,
        metadata_df,
    )


def load_final_model(
    device: torch.device,
) -> tuple[
    CycleAwareRULLSTM,
    dict,
]:

    checkpoint = torch.load(
        FINAL_MODEL_PATH,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "input_size",
        "hidden_size",
        "num_layers",
        "epoch",
        "sensor_columns",
        "window_length",
        "training_engine_count",
    }

    missing_keys = (
        required_keys
        - set(
            checkpoint.keys()
        )
    )

    if missing_keys:
        raise ValueError(
            "Final model checkpoint is missing "
            f"keys: {sorted(missing_keys)}"
        )

    if (
        checkpoint[
            "sensor_columns"
        ]
        != USEFUL_SENSOR_COLUMNS
    ):
        raise ValueError(
            "Final checkpoint sensor order does "
            "not match the locked sensor order."
        )

    if (
        checkpoint[
            "window_length"
        ]
        != WINDOW_LENGTH
    ):
        raise ValueError(
            "Final checkpoint window length does "
            "not match the locked window length."
        )

    if (
        checkpoint[
            "training_engine_count"
        ]
        != 100
    ):
        raise ValueError(
            "Final model was not trained on exactly "
            "100 training engines."
        )

    model = CycleAwareRULLSTM(
        input_size=checkpoint[
            "input_size"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


def generate_predictions(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    sensor_tensor = torch.tensor(
        sensor_sequences,
        dtype=torch.float32,
    )

    cycle_tensor = torch.tensor(
        cycle_features,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        sensor_tensor,
        cycle_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    with torch.no_grad():

        for (
            batch_sequences,
            batch_cycle_features,
        ) in loader:

            batch_sequences = (
                batch_sequences.to(
                    device
                )
            )

            batch_cycle_features = (
                batch_cycle_features.to(
                    device
                )
            )

            batch_predictions = model(
                batch_sequences,
                batch_cycle_features,
            )

            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )

    prediction_array = np.concatenate(
        predictions
    )

    if prediction_array.shape != (
        100,
    ):
        raise ValueError(
            "Expected exactly 100 official-test "
            "predictions."
        )

    if not np.isfinite(
        prediction_array
    ).all():
        raise ValueError(
            "Official-test predictions contain "
            "non-finite values."
        )

    return prediction_array


def main() -> None:

    print(
        "\n======================================"
    )

    print(
        "FD001 OFFICIAL TEST PREDICTION"
    )

    print(
        "======================================"
    )

    protocol = load_locked_protocol()

    validate_final_artifacts()

    print(
        "\nDevelopment lock:"
    )

    print(
        "PASSED"
    )

    print(
        "\nFinal artifact check:"
    )

    print(
        "PASSED"
    )

    device = torch.device(
        "cpu"
    )

    test_df = load_test_data()

    print(
        "\nOfficial test source shape:"
    )

    print(
        test_df.shape
    )

    print(
        "\nOfficial test engines:"
    )

    print(
        test_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nOfficial test RUL loaded:"
    )

    print(
        "NO"
    )

    sensor_scaler = joblib.load(
        FINAL_SENSOR_SCALER_PATH
    )

    cycle_scaler = joblib.load(
        FINAL_CYCLE_SCALER_PATH
    )

    (
        sensor_sequences,
        cycle_features,
        metadata_df,
    ) = build_final_test_inputs(
        test_df,
        sensor_scaler,
        cycle_scaler,
    )

    print(
        "\nOfficial test sequence shape:"
    )

    print(
        sensor_sequences.shape
    )

    print(
        "\nOfficial test cycle feature shape:"
    )

    print(
        cycle_features.shape
    )

    print(
        "\nSensor order:"
    )

    print(
        USEFUL_SENSOR_COLUMNS
    )

    print(
        "\nNaN / infinite input check:"
    )

    print(
        "PASSED"
    )

    model, checkpoint = (
        load_final_model(
            device
        )
    )

    print(
        "\nFinal model checkpoint:"
    )

    print(
        "PASSED"
    )

    print(
        "\nFinal model training engines:"
    )

    print(
        checkpoint[
            "training_engine_count"
        ]
    )

    print(
        "\nFinal model epochs:"
    )

    print(
        checkpoint[
            "epoch"
        ]
    )

    prediction_array = (
        generate_predictions(
            model,
            sensor_sequences,
            cycle_features,
            device,
        )
    )

    prediction_df = (
        metadata_df.copy()
    )

    prediction_df[
        "predicted_rul"
    ] = prediction_array

    if (
        prediction_df[
            "engine_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate official-test engine "
            "predictions were produced."
        )

    if len(
        prediction_df
    ) != 100:
        raise ValueError(
            "Unscored prediction table must "
            "contain exactly 100 rows."
        )

    UNSCORED_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        UNSCORED_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "\nPredictions generated:"
    )

    print(
        len(
            prediction_df
        )
    )

    print(
        "\nOne prediction per engine:"
    )

    print(
        "PASSED"
    )

    print(
        "\nFirst 5 unscored predictions:"
    )

    print(
        prediction_df.head()
    )

    print(
        "\nUnscored predictions saved to:"
    )

    print(
        UNSCORED_PREDICTIONS_PATH
    )

    print(
        "\n======================================"
    )

    print(
        "OFFICIAL TEST PREDICTIONS FROZEN"
    )

    print(
        "======================================"
    )

    print(
        "\nOfficial test RUL loaded:"
    )

    print(
        "NO"
    )


if __name__ == "__main__":
    main()