from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    SAVED_ARTIFACTS_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.data.load_data import load_train_data
from src.data.split_data import get_development_split
from src.features.rul import add_training_rul


WINDOW_LENGTH = 30
STRIDE = 1

TRAIN_SEQUENCE_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_train_sequences.npy"
)

VALIDATION_SEQUENCE_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_validation_sequences.npy"
)

TRAIN_SEQUENCE_TARGET_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_train_sequence_targets.npy"
)

VALIDATION_SEQUENCE_TARGET_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_validation_sequence_targets.npy"
)

TRAIN_SEQUENCE_METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_train_sequence_metadata.csv"
)

VALIDATION_SEQUENCE_METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_validation_sequence_metadata.csv"
)

SEQUENCE_SCALER_FILE = (
    SAVED_ARTIFACTS_DIR
    / "fd001_sequence_sensor_scaler.joblib"
)

def load_sequence_source_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    train_df = load_train_data()

    train_with_rul_df = add_training_rul(
        train_df
    )

    (
        development_train_df,
        development_validation_df,
        _,
    ) = get_development_split(
        train_with_rul_df
    )

    return (
        development_train_df,
        development_validation_df,
    )


def fit_sensor_scaler(
    development_train_df: pd.DataFrame,
) -> StandardScaler:

    scaler = StandardScaler()

    scaler.fit(
        development_train_df[
            USEFUL_SENSOR_COLUMNS
        ]
    )

    return scaler


def apply_sensor_scaling(
    dataframe: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:

    scaled_df = dataframe.copy()

    scaled_df[
        USEFUL_SENSOR_COLUMNS
    ] = scaler.transform(
        dataframe[
            USEFUL_SENSOR_COLUMNS
        ]
    )

    return scaled_df


def build_sequences(
    dataframe: pd.DataFrame,
    window_length: int = WINDOW_LENGTH,
    stride: int = STRIDE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:

    sequences = []
    targets = []
    metadata_rows = []

    for engine_id, engine_df in dataframe.groupby(
        "engine_id",
        sort=True,
    ):

        engine_df = (
            engine_df
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        number_of_rows = len(engine_df)

        if number_of_rows < window_length:
            continue

        for start_index in range(
            0,
            number_of_rows - window_length + 1,
            stride,
        ):

            end_index = (
                start_index
                + window_length
                - 1
            )

            sequence = (
                engine_df.loc[
                    start_index:end_index,
                    USEFUL_SENSOR_COLUMNS,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            target = np.float32(
                engine_df.loc[
                    end_index,
                    "rul",
                ]
            )

            ending_cycle = int(
                engine_df.loc[
                    end_index,
                    "cycle",
                ]
            )

            sequences.append(sequence)

            targets.append(target)

            metadata_rows.append(
                {
                    "engine_id": int(engine_id),
                    "cycle": ending_cycle,
                    "rul": float(target),
                }
            )

    sequences_array = np.stack(
        sequences
    )

    targets_array = np.asarray(
        targets,
        dtype=np.float32,
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    return (
        sequences_array,
        targets_array,
        metadata_df,
    )


def calculate_expected_sequence_count(
    dataframe: pd.DataFrame,
    window_length: int = WINDOW_LENGTH,
    stride: int = STRIDE,
) -> int:

    expected_count = 0

    for _, engine_df in dataframe.groupby(
        "engine_id"
    ):

        number_of_rows = len(engine_df)

        if number_of_rows < window_length:
            continue

        engine_sequence_count = (
            (
                number_of_rows
                - window_length
            )
            // stride
            + 1
        )

        expected_count += (
            engine_sequence_count
        )

    return expected_count


def validate_sequence_alignment(
    source_df: pd.DataFrame,
    targets: np.ndarray,
    metadata_df: pd.DataFrame,
) -> None:

    source_targets = (
        source_df[
            [
                "engine_id",
                "cycle",
                "rul",
            ]
        ]
        .copy()
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
    )

    metadata_with_target = (
        metadata_df
        .copy()
        .rename(
            columns={
                "rul":
                    "metadata_rul"
            }
        )
    )

    metadata_with_target[
        "array_target"
    ] = targets

    alignment_df = (
        metadata_with_target
        .merge(
            source_targets,
            on=[
                "engine_id",
                "cycle",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if alignment_df["rul"].isna().any():
        raise ValueError(
            "Sequence metadata contains an "
            "engine/cycle pair not found in the "
            "source table."
        )

    if not np.allclose(
        alignment_df["metadata_rul"],
        alignment_df["rul"],
    ):
        raise ValueError(
            "Metadata RUL does not align with "
            "source-table RUL."
        )

    if not np.allclose(
        alignment_df["array_target"],
        alignment_df["rul"],
    ):
        raise ValueError(
            "Sequence target array does not align "
            "with source-table RUL."
        ) 

    
def validate_sequence_data(
    sequences: np.ndarray,
    targets: np.ndarray,
    metadata_df: pd.DataFrame,
    window_length: int = WINDOW_LENGTH,
) -> None:
 

    if sequences.ndim != 3:
        raise ValueError(
            "Sequence array must have exactly "
            "three dimensions."
        )

    if sequences.shape[1] != window_length:
        raise ValueError(
            "Unexpected sequence length."
        )

    if sequences.shape[2] != len(
        USEFUL_SENSOR_COLUMNS
    ):
        raise ValueError(
            "Unexpected number of sensor channels."
        )

    number_of_samples = sequences.shape[0]

    if len(targets) != number_of_samples:
        raise ValueError(
            "Targets do not align with sequences."
        )

    if len(metadata_df) != number_of_samples:
        raise ValueError(
            "Metadata does not align with sequences."
        )

    if not np.isfinite(sequences).all():
        raise ValueError(
            "Sequence data contains non-finite values."
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            "Targets contain non-finite values."
        )
    
    if metadata_df[
        [
            "engine_id",
            "cycle",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate engine/cycle prediction "
            "points found in sequence metadata."
        )
    
def save_sequence_artifacts(
    x_train_sequences: np.ndarray,
    y_train_sequences: np.ndarray,
    train_metadata_df: pd.DataFrame,
    x_validation_sequences: np.ndarray,
    y_validation_sequences: np.ndarray,
    validation_metadata_df: pd.DataFrame,
    scaler: StandardScaler,
) -> None:

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAVED_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        TRAIN_SEQUENCE_FILE,
        x_train_sequences,
    )

    np.save(
        VALIDATION_SEQUENCE_FILE,
        x_validation_sequences,
    )

    np.save(
        TRAIN_SEQUENCE_TARGET_FILE,
        y_train_sequences,
    )

    np.save(
        VALIDATION_SEQUENCE_TARGET_FILE,
        y_validation_sequences,
    )

    train_metadata_df.to_csv(
        TRAIN_SEQUENCE_METADATA_FILE,
        index=False,
    )

    validation_metadata_df.to_csv(
        VALIDATION_SEQUENCE_METADATA_FILE,
        index=False,
    )

    joblib.dump(
        scaler,
        SEQUENCE_SCALER_FILE,
    )

def main() -> None:

    (
        development_train_df,
        development_validation_df,
    ) = load_sequence_source_tables()

    scaler = fit_sensor_scaler(
        development_train_df
    )

    scaled_train_df = apply_sensor_scaling(
        development_train_df,
        scaler,
    )

    scaled_validation_df = apply_sensor_scaling(
        development_validation_df,
        scaler,
    )

    (
        x_train_sequences,
        y_train_sequences,
        train_metadata_df,
    ) = build_sequences(
        scaled_train_df
    )

    (
        x_validation_sequences,
        y_validation_sequences,
        validation_metadata_df,
    ) = build_sequences(
        scaled_validation_df
    )

    validate_sequence_data(
        x_train_sequences,
        y_train_sequences,
        train_metadata_df,
    )

    validate_sequence_data(
        x_validation_sequences,
        y_validation_sequences,
        validation_metadata_df,
    )


    expected_train_count = (
        calculate_expected_sequence_count(
            development_train_df
        )
    )

    expected_validation_count = (
        calculate_expected_sequence_count(
            development_validation_df
        )
    )

    if (
        len(x_train_sequences)
        != expected_train_count
    ):
        raise ValueError(
            "Training sequence count does not "
            "match expected count."
        )

    if (
        len(x_validation_sequences)
        != expected_validation_count
    ):
        raise ValueError(
            "Validation sequence count does not "
            "match expected count."
        )

    validate_sequence_alignment(
        development_train_df,
        y_train_sequences,
        train_metadata_df,
    )

    validate_sequence_alignment(
        development_validation_df,
        y_validation_sequences,
        validation_metadata_df,
    )

    save_sequence_artifacts(
        x_train_sequences,
        y_train_sequences,
        train_metadata_df,
        x_validation_sequences,
        y_validation_sequences,
        validation_metadata_df,
        scaler,
    )


    print(
        "\n======================================"
    )
    print(
        "FD001 SEQUENCE PREPARATION CHECK"
    )
    print(
        "======================================"
    )

    print(
        "\nWindow length:"
    )
    print(
        WINDOW_LENGTH
    )

    print(
        "\nStride:"
    )
    print(
        STRIDE
    )

    print(
        "\nInput sensor channels:"
    )
    print(
        len(USEFUL_SENSOR_COLUMNS)
    )

    print(
        "\nDevelopment-training source shape:"
    )
    print(
        development_train_df.shape
    )

    print(
        "\nDevelopment-validation source shape:"
    )
    print(
        development_validation_df.shape
    )

    print(
        "\nTraining sequence shape:"
    )
    print(
        x_train_sequences.shape
    )

    print(
        "\nValidation sequence shape:"
    )
    print(
        x_validation_sequences.shape
    )

    print(
        "\nTraining target shape:"
    )
    print(
        y_train_sequences.shape
    )

    print(
        "\nValidation target shape:"
    )
    print(
        y_validation_sequences.shape
    )

    print(
        "\nFirst training metadata rows:"
    )
    print(
        train_metadata_df.head()
    )

    print(
        "\nFirst validation metadata rows:"
    )
    print(
        validation_metadata_df.head()
    )

    print(
        "\nTraining scaled sensor means:"
    )
    print(
        scaled_train_df[
            USEFUL_SENSOR_COLUMNS
        ]
        .mean()
        .round(4)
    )

    print(
        "\nSequence validation passed."
    )

    print(
        "\nExpected training sequences:"
    )
    print(
        expected_train_count
    )

    print(
        "\nExpected validation sequences:"
    )
    print(
        expected_validation_count
    )

    print(
        "\nAlignment validation passed."
    )

    print(
        "\nSequence artifacts saved."
    )

    print(
        "\nTraining sequences:"
    )
    print(
        TRAIN_SEQUENCE_FILE
    )

    print(
        "\nValidation sequences:"
    )
    print(
        VALIDATION_SEQUENCE_FILE
    )

    print(
        "\nSequence scaler:"
    )
    print(
        SEQUENCE_SCALER_FILE
    )

if __name__ == "__main__":
    main()