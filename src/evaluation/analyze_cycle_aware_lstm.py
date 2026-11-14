from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.evaluation.rul_metrics import (
    build_prediction_table,
    calculate_engine_metrics,
    calculate_regression_metrics,
    summarize_engine_metrics,
)
from src.modeling.cycle_aware_lstm_model import (
    CycleAwareRULLSTM,
)
from src.modeling.cycle_aware_sequence_dataset import (
    CycleAwareRULSequenceDataset,
)


VALIDATION_SEQUENCES_PATH = Path(
    "data/processed/fd001_validation_sequences.npy"
)

VALIDATION_TARGETS_PATH = Path(
    "data/processed/fd001_validation_sequence_targets.npy"
)

VALIDATION_METADATA_PATH = Path(
    "data/processed/fd001_validation_sequence_metadata.csv"
)


MODEL_PATH = Path(
    "models/saved_artifacts/"
    "fd001_cycle_aware_lstm_best.pt"
)

CYCLE_SCALER_PATH = Path(
    "models/saved_artifacts/"
    "fd001_cycle_feature_scaler.joblib"
)


SENSOR_ONLY_PREDICTIONS_PATH = Path(
    "outputs/lstm_validation_predictions.csv"
)

RF_PREDICTIONS_PATH = Path(
    "outputs/rf_shared_validation_predictions.csv"
)


CYCLE_AWARE_PREDICTIONS_PATH = Path(
    "outputs/"
    "cycle_aware_lstm_validation_predictions.csv"
)

CYCLE_AWARE_ENGINE_METRICS_PATH = Path(
    "outputs/"
    "cycle_aware_lstm_validation_engine_metrics.csv"
)

CYCLE_AWARE_RUL_SLICES_PATH = Path(
    "outputs/"
    "cycle_aware_lstm_validation_rul_slices.csv"
)

DEVELOPMENT_COMPARISON_PATH = Path(
    "outputs/"
    "development_model_comparison.csv"
)

ENGINE_COMPARISON_PATH = Path(
    "outputs/"
    "development_model_engine_comparison.csv"
)


BATCH_SIZE = 64


def generate_predictions(
    model: CycleAwareRULLSTM,
    dataset: CycleAwareRULSequenceDataset,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():

        for (
            sensor_sequences,
            cycle_features,
            targets,
        ) in loader:

            sensor_sequences = (
                sensor_sequences.to(
                    device
                )
            )

            cycle_features = (
                cycle_features.to(
                    device
                )
            )

            predictions = model(
                sensor_sequences,
                cycle_features,
            )

            all_targets.append(
                targets.numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    return (
        np.concatenate(
            all_targets
        ),
        np.concatenate(
            all_predictions
        ),
    )


def calculate_rul_slices(
    prediction_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    band_definitions = [
        (
            "near_failure_0_30",
            (
                prediction_df[
                    "actual_rul"
                ]
                <= 30
            ),
        ),
        (
            "medium_rul_31_100",
            (
                (
                    prediction_df[
                        "actual_rul"
                    ]
                    > 30
                )
                & (
                    prediction_df[
                        "actual_rul"
                    ]
                    <= 100
                )
            ),
        ),
        (
            "high_rul_above_100",
            (
                prediction_df[
                    "actual_rul"
                ]
                > 100
            ),
        ),
    ]

    rows: list[
        dict[str, float | int | str]
    ] = []

    for (
        band_name,
        band_mask,
    ) in band_definitions:

        band_df = prediction_df.loc[
            band_mask
        ]

        overprediction_df = (
            band_df.loc[
                band_df[
                    "error"
                ]
                > 0
            ]
        )

        if len(overprediction_df) > 0:

            mean_overprediction = float(
                overprediction_df[
                    "error"
                ].mean()
            )

        else:

            mean_overprediction = float(
                "nan"
            )

        rows.append(
            {
                "model": model_name,
                "rul_band": band_name,
                "row_count": int(
                    len(band_df)
                ),
                "engine_count": int(
                    band_df[
                        "engine_id"
                    ].nunique()
                ),
                "mae": float(
                    band_df[
                        "absolute_error"
                    ].mean()
                ),
                "rmse": float(
                    np.sqrt(
                        band_df[
                            "squared_error"
                        ].mean()
                    )
                ),
                "mean_signed_error": float(
                    band_df[
                        "error"
                    ].mean()
                ),
                "overprediction_fraction": float(
                    (
                        band_df[
                            "error"
                        ]
                        > 0
                    ).mean()
                ),
                "mean_overprediction_when_over": (
                    mean_overprediction
                ),
                "overprediction_gt20_fraction": float(
                    (
                        band_df[
                            "error"
                        ]
                        > 20
                    ).mean()
                ),
                "p95_absolute_error": float(
                    band_df[
                        "absolute_error"
                    ].quantile(
                        0.95
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def load_existing_prediction_table(
    path: Path,
    model_name: str,
) -> pd.DataFrame:

    prediction_df = pd.read_csv(
        path
    )

    required_columns = [
        "engine_id",
        "cycle",
        "actual_rul",
        "predicted_rul",
        "error",
        "absolute_error",
        "squared_error",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{model_name} prediction file "
            f"is missing: {missing_columns}"
        )

    return prediction_df


def summarize_model(
    model_name: str,
    prediction_df: pd.DataFrame,
) -> dict[str, float | str]:

    metrics = calculate_regression_metrics(
        prediction_df[
            "actual_rul"
        ].to_numpy(),
        prediction_df[
            "predicted_rul"
        ].to_numpy(),
    )

    engine_metrics_df = (
        calculate_engine_metrics(
            prediction_df
        )
    )

    engine_summary = (
        summarize_engine_metrics(
            engine_metrics_df
        )
    )

    high_rul_df = prediction_df.loc[
        prediction_df[
            "actual_rul"
        ]
        > 100
    ]

    near_failure_df = prediction_df.loc[
        prediction_df[
            "actual_rul"
        ]
        <= 30
    ]

    medium_rul_df = prediction_df.loc[
        (
            prediction_df[
                "actual_rul"
            ]
            > 30
        )
        & (
            prediction_df[
                "actual_rul"
            ]
            <= 100
        )
    ]

    return {
        "model": model_name,
        "mae": float(
            metrics[
                "mae"
            ]
        ),
        "rmse": float(
            metrics[
                "rmse"
            ]
        ),
        "mean_engine_mae": float(
            engine_summary[
                "mean_engine_mae"
            ]
        ),
        "median_engine_mae": float(
            engine_summary[
                "median_engine_mae"
            ]
        ),
        "mean_engine_rmse": float(
            engine_summary[
                "mean_engine_rmse"
            ]
        ),
        "near_failure_mae": float(
            near_failure_df[
                "absolute_error"
            ].mean()
        ),
        "medium_rul_mae": float(
            medium_rul_df[
                "absolute_error"
            ].mean()
        ),
        "high_rul_mae": float(
            high_rul_df[
                "absolute_error"
            ].mean()
        ),
        "high_rul_mean_signed_error": float(
            high_rul_df[
                "error"
            ].mean()
        ),
    }


def calculate_per_engine_mae(
    prediction_df: pd.DataFrame,
    model_prefix: str,
) -> pd.DataFrame:

    rows: list[
        dict[str, float | int]
    ] = []

    for (
        engine_id,
        engine_df,
    ) in prediction_df.groupby(
        "engine_id",
        sort=True,
    ):

        rows.append(
            {
                "engine_id": int(
                    engine_id
                ),
                f"{model_prefix}_mae": float(
                    engine_df[
                        "absolute_error"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "\n======================================"
    )
    print(
        "FD001 CYCLE-AWARE LSTM ANALYSIS"
    )
    print(
        "======================================"
    )

    device = torch.device(
        "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
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

    print(
        "\nLoaded checkpoint:"
    )

    print(
        f"Epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved validation MAE: "
        f"{checkpoint['validation_mae']:.4f}"
    )

    x_validation_sequences = np.load(
        VALIDATION_SEQUENCES_PATH
    )

    y_validation_sequences = np.load(
        VALIDATION_TARGETS_PATH
    )

    validation_metadata_df = pd.read_csv(
        VALIDATION_METADATA_PATH
    )

    if (
        len(
            validation_metadata_df
        )
        != len(
            x_validation_sequences
        )
    ):

        raise ValueError(
            "Validation metadata and sequence "
            "row counts do not match."
        )

    if not np.allclose(
        validation_metadata_df[
            "rul"
        ].to_numpy(),
        y_validation_sequences,
    ):

        raise ValueError(
            "Validation metadata RUL does not "
            "match validation targets."
        )

    cycle_scaler = joblib.load(
        CYCLE_SCALER_PATH
    )

    validation_cycle_features = (
        cycle_scaler.transform(
            validation_metadata_df[
                ["cycle"]
            ].to_numpy()
        )
        .astype(
            np.float32
        )
    )

    dataset = (
        CycleAwareRULSequenceDataset(
            x_validation_sequences,
            validation_cycle_features,
            y_validation_sequences,
        )
    )

    (
        validation_targets,
        validation_predictions,
    ) = generate_predictions(
        model,
        dataset,
        device,
    )

    metrics = calculate_regression_metrics(
        validation_targets,
        validation_predictions,
    )

    prediction_df = build_prediction_table(
        validation_metadata_df[
            [
                "engine_id",
                "cycle",
            ]
        ],
        validation_targets,
        validation_predictions,
    )

    engine_metrics_df = (
        calculate_engine_metrics(
            prediction_df
        )
    )

    engine_summary = (
        summarize_engine_metrics(
            engine_metrics_df
        )
    )

    rul_slices_df = calculate_rul_slices(
        prediction_df,
        "cycle_aware_lstm",
    )

    print(
        "\n======================================"
    )
    print(
        "CYCLE-AWARE VALIDATION RESULTS"
    )
    print(
        "======================================"
    )

    print(
        f"\nRows: "
        f"{len(prediction_df)}"
    )

    print(
        f"Engines: "
        f"{prediction_df['engine_id'].nunique()}"
    )

    print(
        f"MAE: "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"Mean engine MAE: "
        f"{engine_summary['mean_engine_mae']:.4f}"
    )

    print(
        f"Median engine MAE: "
        f"{engine_summary['median_engine_mae']:.4f}"
    )

    print(
        f"Mean engine RMSE: "
        f"{engine_summary['mean_engine_rmse']:.4f}"
    )

    print(
        "\nCycle-aware RUL slices:"
    )

    print(
        rul_slices_df.to_string(
            index=False
        )
    )

    sensor_only_df = (
        load_existing_prediction_table(
            SENSOR_ONLY_PREDICTIONS_PATH,
            "Sensor-only LSTM",
        )
    )

    rf_df = (
        load_existing_prediction_table(
            RF_PREDICTIONS_PATH,
            "Random Forest",
        )
    )

    if len(sensor_only_df) != len(
        prediction_df
    ):

        raise ValueError(
            "Sensor-only and cycle-aware "
            "prediction counts do not match."
        )

    if len(rf_df) != len(
        prediction_df
    ):

        raise ValueError(
            "RF and cycle-aware prediction "
            "counts do not match."
        )

    cycle_keys = prediction_df[
        [
            "engine_id",
            "cycle",
        ]
    ].sort_values(
        [
            "engine_id",
            "cycle",
        ]
    ).reset_index(
        drop=True
    )

    sensor_keys = sensor_only_df[
        [
            "engine_id",
            "cycle",
        ]
    ].sort_values(
        [
            "engine_id",
            "cycle",
        ]
    ).reset_index(
        drop=True
    )

    rf_keys = rf_df[
        [
            "engine_id",
            "cycle",
        ]
    ].sort_values(
        [
            "engine_id",
            "cycle",
        ]
    ).reset_index(
        drop=True
    )

    if not cycle_keys.equals(
        sensor_keys
    ):

        raise ValueError(
            "Cycle-aware and sensor-only "
            "prediction keys do not match."
        )

    if not cycle_keys.equals(
        rf_keys
    ):

        raise ValueError(
            "Cycle-aware and RF prediction "
            "keys do not match."
        )

    model_summaries = [
        summarize_model(
            "cycle_aware_lstm",
            prediction_df,
        ),
        summarize_model(
            "sensor_only_lstm",
            sensor_only_df,
        ),
        summarize_model(
            "random_forest_depth12",
            rf_df,
        ),
    ]

    comparison_df = pd.DataFrame(
        model_summaries
    )

    print(
        "\n======================================"
    )
    print(
        "DEVELOPMENT MODEL COMPARISON"
    )
    print(
        "======================================"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    cycle_engine_df = (
        calculate_per_engine_mae(
            prediction_df,
            "cycle_aware",
        )
    )

    sensor_engine_df = (
        calculate_per_engine_mae(
            sensor_only_df,
            "sensor_only",
        )
    )

    rf_engine_df = (
        calculate_per_engine_mae(
            rf_df,
            "rf",
        )
    )

    engine_comparison_df = (
        cycle_engine_df
        .merge(
            sensor_engine_df,
            on="engine_id",
            validate="one_to_one",
        )
        .merge(
            rf_engine_df,
            on="engine_id",
            validate="one_to_one",
        )
    )

    engine_comparison_df[
        "sensor_only_minus_cycle_aware"
    ] = (
        engine_comparison_df[
            "sensor_only_mae"
        ]
        - engine_comparison_df[
            "cycle_aware_mae"
        ]
    )

    engine_comparison_df[
        "rf_minus_cycle_aware"
    ] = (
        engine_comparison_df[
            "rf_mae"
        ]
        - engine_comparison_df[
            "cycle_aware_mae"
        ]
    )

    cycle_beats_sensor = int(
        (
            engine_comparison_df[
                "sensor_only_minus_cycle_aware"
            ]
            > 0
        ).sum()
    )

    sensor_beats_cycle = int(
        (
            engine_comparison_df[
                "sensor_only_minus_cycle_aware"
            ]
            < 0
        ).sum()
    )

    cycle_beats_rf = int(
        (
            engine_comparison_df[
                "rf_minus_cycle_aware"
            ]
            > 0
        ).sum()
    )

    rf_beats_cycle = int(
        (
            engine_comparison_df[
                "rf_minus_cycle_aware"
            ]
            < 0
        ).sum()
    )

    print(
        "\n======================================"
    )
    print(
        "PER-ENGINE CONSISTENCY"
    )
    print(
        "======================================"
    )

    print(
        "\nCycle-aware beats sensor-only "
        "LSTM on engines:"
    )

    print(
        cycle_beats_sensor
    )

    print(
        "\nSensor-only LSTM beats "
        "cycle-aware on engines:"
    )

    print(
        sensor_beats_cycle
    )

    print(
        "\nCycle-aware beats RF "
        "on engines:"
    )

    print(
        cycle_beats_rf
    )

    print(
        "\nRF beats cycle-aware "
        "on engines:"
    )

    print(
        rf_beats_cycle
    )

    CYCLE_AWARE_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        CYCLE_AWARE_PREDICTIONS_PATH,
        index=False,
    )

    engine_metrics_df.to_csv(
        CYCLE_AWARE_ENGINE_METRICS_PATH,
        index=False,
    )

    rul_slices_df.to_csv(
        CYCLE_AWARE_RUL_SLICES_PATH,
        index=False,
    )

    comparison_df.to_csv(
        DEVELOPMENT_COMPARISON_PATH,
        index=False,
    )

    engine_comparison_df.to_csv(
        ENGINE_COMPARISON_PATH,
        index=False,
    )

    print(
        "\n======================================"
    )
    print(
        "SAVED OUTPUTS"
    )
    print(
        "======================================"
    )

    print(
        CYCLE_AWARE_PREDICTIONS_PATH
    )

    print(
        CYCLE_AWARE_ENGINE_METRICS_PATH
    )

    print(
        CYCLE_AWARE_RUL_SLICES_PATH
    )

    print(
        DEVELOPMENT_COMPARISON_PATH
    )

    print(
        ENGINE_COMPARISON_PATH
    )


if __name__ == "__main__":
    main()