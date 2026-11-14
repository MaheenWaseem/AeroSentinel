from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.rul_metrics import (
    build_prediction_table,
    calculate_engine_metrics,
    calculate_regression_metrics,
    summarize_engine_metrics,
)
from src.modeling.lstm_model import (
    RULLSTM,
)
from src.modeling.sequence_dataset import (
    RULSequenceDataset,
)


BATCH_SIZE = 64

VALIDATION_SEQUENCES_PATH = Path(
    "data/processed/fd001_validation_sequences.npy"
)

VALIDATION_TARGETS_PATH = Path(
    "data/processed/fd001_validation_sequence_targets.npy"
)

VALIDATION_METADATA_PATH = Path(
    "data/processed/fd001_validation_sequence_metadata.csv"
)

BEST_MODEL_PATH = Path(
    "models/saved_artifacts/fd001_lstm_best.pt"
)

CLASSICAL_BENCHMARK_PREDICTIONS_PATH = Path(
    "outputs/classical_benchmark_predictions.csv"
)


LSTM_PREDICTIONS_PATH = Path(
    "outputs/lstm_validation_predictions.csv"
)

LSTM_ENGINE_METRICS_PATH = Path(
    "outputs/lstm_validation_engine_metrics.csv"
)

LSTM_RUL_SLICES_PATH = Path(
    "outputs/lstm_validation_rul_slices.csv"
)

RF_SHARED_PREDICTIONS_PATH = Path(
    "outputs/rf_shared_validation_predictions.csv"
)

RF_SHARED_ENGINE_METRICS_PATH = Path(
    "outputs/rf_shared_validation_engine_metrics.csv"
)

RF_SHARED_RUL_SLICES_PATH = Path(
    "outputs/rf_shared_validation_rul_slices.csv"
)

SHARED_COMPARISON_PATH = Path(
    "outputs/lstm_vs_rf_shared_comparison.csv"
)


ORIGINAL_RF_ROW_COUNT = 3970
ORIGINAL_RF_MAE = 23.0152
ORIGINAL_RF_RMSE = 31.1843
ORIGINAL_RF_MEAN_ENGINE_MAE = 23.3942


def load_best_lstm(
    device: torch.device,
) -> tuple[
    RULLSTM,
    dict,
]:

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model = RULLSTM(
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


def generate_lstm_predictions(
    model: RULLSTM,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            predictions = model(
                inputs
            )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

    return (
        y_true,
        y_pred,
    )


def calculate_rul_slice_metrics(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:

    prediction_df = (
        prediction_df.copy()
    )

    prediction_df[
        "rul_band"
    ] = np.select(
        [
            prediction_df[
                "actual_rul"
            ] <= 30,
            (
                prediction_df[
                    "actual_rul"
                ] >= 31
            )
            & (
                prediction_df[
                    "actual_rul"
                ] <= 100
            ),
        ],
        [
            "near_failure_0_30",
            "medium_rul_31_100",
        ],
        default="high_rul_above100",
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    band_order = [
        "near_failure_0_30",
        "medium_rul_31_100",
        "high_rul_above100",
    ]

    for rul_band in band_order:

        band_df = prediction_df[
            prediction_df[
                "rul_band"
            ]
            == rul_band
        ]

        if band_df.empty:
            continue

        overprediction_df = band_df[
            band_df[
                "error"
            ] > 0
        ]

        if overprediction_df.empty:

            mean_overprediction_when_over = (
                0.0
            )

        else:

            mean_overprediction_when_over = (
                overprediction_df[
                    "error"
                ]
                .mean()
            )

        mae = (
            band_df[
                "absolute_error"
            ]
            .mean()
        )

        rmse = (
            band_df[
                "squared_error"
            ]
            .mean()
            ** 0.5
        )

        mean_signed_error = (
            band_df[
                "error"
            ]
            .mean()
        )

        overprediction_fraction = (
            (
                band_df[
                    "error"
                ]
                > 0
            )
            .mean()
        )

        overprediction_gt20_fraction = (
            (
                band_df[
                    "error"
                ]
                > 20
            )
            .mean()
        )

        p95_absolute_error = (
            band_df[
                "absolute_error"
            ]
            .quantile(
                0.95
            )
        )

        rows.append(
            {
                "rul_band": rul_band,
                "row_count": len(
                    band_df
                ),
                "engine_count": (
                    band_df[
                        "engine_id"
                    ]
                    .nunique()
                ),
                "mae": float(
                    mae
                ),
                "rmse": float(
                    rmse
                ),
                "mean_signed_error": float(
                    mean_signed_error
                ),
                "overprediction_fraction": float(
                    overprediction_fraction
                ),
                "mean_overprediction_when_over": float(
                    mean_overprediction_when_over
                ),
                "overprediction_gt20_fraction": float(
                    overprediction_gt20_fraction
                ),
                "p95_absolute_error": float(
                    p95_absolute_error
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_shared_rf_predictions(
    lstm_prediction_df: pd.DataFrame,
    rf_prediction_df: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = {
        "engine_id",
        "cycle",
        "actual_rul",
        "predicted_rul",
    }

    missing_columns = (
        required_columns
        - set(
            rf_prediction_df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Classical benchmark prediction file "
            f"is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if rf_prediction_df.duplicated(
        subset=[
            "engine_id",
            "cycle",
        ]
    ).any():

        raise ValueError(
            "Duplicate engine/cycle pairs found "
            "in classical benchmark predictions."
        )

    shared_keys = (
        lstm_prediction_df[
            [
                "engine_id",
                "cycle",
                "actual_rul",
            ]
        ]
        .copy()
    )

    rf_shared_df = (
        shared_keys.merge(
            rf_prediction_df[
                [
                    "engine_id",
                    "cycle",
                    "actual_rul",
                    "predicted_rul",
                ]
            ],
            on=[
                "engine_id",
                "cycle",
            ],
            how="left",
            validate="one_to_one",
            suffixes=(
                "_lstm",
                "_rf",
            ),
        )
    )

    if rf_shared_df[
        "predicted_rul"
    ].isna().any():

        raise ValueError(
            "Some LSTM validation prediction points "
            "were not found in the RF benchmark."
        )

    actual_rul_matches = np.allclose(
        rf_shared_df[
            "actual_rul_lstm"
        ].to_numpy(),
        rf_shared_df[
            "actual_rul_rf"
        ].to_numpy(),
    )

    if not actual_rul_matches:

        raise ValueError(
            "Actual RUL values do not match "
            "between LSTM and RF shared rows."
        )

    shared_prediction_df = pd.DataFrame(
        {
            "engine_id": rf_shared_df[
                "engine_id"
            ],
            "cycle": rf_shared_df[
                "cycle"
            ],
            "actual_rul": rf_shared_df[
                "actual_rul_lstm"
            ],
            "predicted_rul": rf_shared_df[
                "predicted_rul"
            ],
        }
    )

    shared_prediction_df[
        "error"
    ] = (
        shared_prediction_df[
            "predicted_rul"
        ]
        - shared_prediction_df[
            "actual_rul"
        ]
    )

    shared_prediction_df[
        "absolute_error"
    ] = (
        shared_prediction_df[
            "error"
        ]
        .abs()
    )

    shared_prediction_df[
        "squared_error"
    ] = (
        shared_prediction_df[
            "error"
        ]
        ** 2
    )

    return shared_prediction_df


def build_model_summary(
    model_name: str,
    prediction_df: pd.DataFrame,
    engine_metrics_df: pd.DataFrame,
) -> dict[str, float | int | str]:

    overall_metrics = (
        calculate_regression_metrics(
            prediction_df[
                "actual_rul"
            ].to_numpy(),
            prediction_df[
                "predicted_rul"
            ].to_numpy(),
        )
    )

    engine_summary = (
        summarize_engine_metrics(
            engine_metrics_df
        )
    )

    return {
        "model": model_name,
        "row_count": len(
            prediction_df
        ),
        "engine_count": (
            prediction_df[
                "engine_id"
            ]
            .nunique()
        ),
        "mae": overall_metrics[
            "mae"
        ],
        "rmse": overall_metrics[
            "rmse"
        ],
        "mean_engine_mae": (
            engine_summary[
                "mean_engine_mae"
            ]
        ),
        "median_engine_mae": (
            engine_summary[
                "median_engine_mae"
            ]
        ),
        "mean_engine_rmse": (
            engine_summary[
                "mean_engine_rmse"
            ]
        ),
    }


def main() -> None:

    print(
        "\n======================================"
    )
    print(
        "FD001 LSTM RESULT ANALYSIS"
    )
    print(
        "======================================"
    )

    device = torch.device(
        "cpu"
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

    if len(
        x_validation_sequences
    ) != len(
        validation_metadata_df
    ):

        raise ValueError(
            "Validation sequence and metadata "
            "row counts do not match."
        )

    if len(
        y_validation_sequences
    ) != len(
        validation_metadata_df
    ):

        raise ValueError(
            "Validation target and metadata "
            "row counts do not match."
        )

    metadata_targets = (
        validation_metadata_df[
            "rul"
        ]
        .to_numpy()
    )

    if not np.allclose(
        y_validation_sequences,
        metadata_targets,
    ):

        raise ValueError(
            "Validation targets do not align "
            "with validation metadata."
        )

    validation_dataset = (
        RULSequenceDataset(
            x_validation_sequences,
            y_validation_sequences,
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model, checkpoint = (
        load_best_lstm(
            device
        )
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

    print(
        f"Input size: "
        f"{checkpoint['input_size']}"
    )

    print(
        f"Hidden size: "
        f"{checkpoint['hidden_size']}"
    )

    print(
        f"Number of layers: "
        f"{checkpoint['num_layers']}"
    )

    (
        validation_targets,
        validation_predictions,
    ) = generate_lstm_predictions(
        model,
        validation_loader,
        device,
    )

    lstm_prediction_df = (
        build_prediction_table(
            validation_metadata_df,
            validation_targets,
            validation_predictions,
        )
    )

    lstm_metrics = (
        calculate_regression_metrics(
            validation_targets,
            validation_predictions,
        )
    )

    lstm_engine_metrics_df = (
        calculate_engine_metrics(
            lstm_prediction_df
        )
    )

    lstm_engine_summary = (
        summarize_engine_metrics(
            lstm_engine_metrics_df
        )
    )

    lstm_rul_slices_df = (
        calculate_rul_slice_metrics(
            lstm_prediction_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "LSTM VALIDATION RESULTS"
    )
    print(
        "======================================"
    )

    print(
        f"\nRows: "
        f"{len(lstm_prediction_df)}"
    )

    print(
        f"Engines: "
        f"{lstm_prediction_df['engine_id'].nunique()}"
    )

    print(
        f"MAE: "
        f"{lstm_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{lstm_metrics['rmse']:.4f}"
    )

    print(
        f"Mean engine MAE: "
        f"{lstm_engine_summary['mean_engine_mae']:.4f}"
    )

    print(
        f"Median engine MAE: "
        f"{lstm_engine_summary['median_engine_mae']:.4f}"
    )

    print(
        f"Mean engine RMSE: "
        f"{lstm_engine_summary['mean_engine_rmse']:.4f}"
    )

    print(
        "\nLSTM RUL slices:"
    )

    print(
        lstm_rul_slices_df.to_string(
            index=False
        )
    )

    rf_prediction_df = pd.read_csv(
        CLASSICAL_BENCHMARK_PREDICTIONS_PATH
    )

    rf_shared_prediction_df = (
        build_shared_rf_predictions(
            lstm_prediction_df,
            rf_prediction_df,
        )
    )

    if len(
        rf_shared_prediction_df
    ) != len(
        lstm_prediction_df
    ):

        raise ValueError(
            "RF and LSTM shared-row counts "
            "do not match."
        )

    rf_shared_metrics = (
        calculate_regression_metrics(
            rf_shared_prediction_df[
                "actual_rul"
            ].to_numpy(),
            rf_shared_prediction_df[
                "predicted_rul"
            ].to_numpy(),
        )
    )

    rf_shared_engine_metrics_df = (
        calculate_engine_metrics(
            rf_shared_prediction_df
        )
    )

    rf_shared_engine_summary = (
        summarize_engine_metrics(
            rf_shared_engine_metrics_df
        )
    )

    rf_shared_rul_slices_df = (
        calculate_rul_slice_metrics(
            rf_shared_prediction_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "ORIGINAL RF HISTORICAL REFERENCE"
    )
    print(
        "======================================"
    )

    print(
        f"\nRows: "
        f"{ORIGINAL_RF_ROW_COUNT}"
    )

    print(
        f"MAE: "
        f"{ORIGINAL_RF_MAE:.4f}"
    )

    print(
        f"RMSE: "
        f"{ORIGINAL_RF_RMSE:.4f}"
    )

    print(
        f"Mean engine MAE: "
        f"{ORIGINAL_RF_MEAN_ENGINE_MAE:.4f}"
    )

    print(
        "\nDo NOT compare this historical "
        "3970-row score directly with "
        "the 3490-row LSTM score."
    )

    print(
        "\n======================================"
    )
    print(
        "RF ON LSTM-ELIGIBLE SHARED ROWS"
    )
    print(
        "======================================"
    )

    print(
        f"\nRows: "
        f"{len(rf_shared_prediction_df)}"
    )

    print(
        f"MAE: "
        f"{rf_shared_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{rf_shared_metrics['rmse']:.4f}"
    )

    print(
        f"Mean engine MAE: "
        f"{rf_shared_engine_summary['mean_engine_mae']:.4f}"
    )

    print(
        f"Median engine MAE: "
        f"{rf_shared_engine_summary['median_engine_mae']:.4f}"
    )

    print(
        f"Mean engine RMSE: "
        f"{rf_shared_engine_summary['mean_engine_rmse']:.4f}"
    )

    print(
        "\nRF shared-row RUL slices:"
    )

    print(
        rf_shared_rul_slices_df.to_string(
            index=False
        )
    )

    lstm_summary = (
        build_model_summary(
            "lstm_sensor_only",
            lstm_prediction_df,
            lstm_engine_metrics_df,
        )
    )

    rf_summary = (
        build_model_summary(
            "random_forest_depth12",
            rf_shared_prediction_df,
            rf_shared_engine_metrics_df,
        )
    )

    comparison_df = pd.DataFrame(
        [
            lstm_summary,
            rf_summary,
        ]
    )

    print(
        "\n======================================"
    )
    print(
        "FAIR SHARED-ROW COMPARISON"
    )
    print(
        "======================================"
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    mae_difference = (
        rf_shared_metrics[
            "mae"
        ]
        - lstm_metrics[
            "mae"
        ]
    )

    rmse_difference = (
        rf_shared_metrics[
            "rmse"
        ]
        - lstm_metrics[
            "rmse"
        ]
    )

    print(
        "\nRF MAE minus LSTM MAE:"
    )

    print(
        f"{mae_difference:.4f}"
    )

    print(
        "\nRF RMSE minus LSTM RMSE:"
    )

    print(
        f"{rmse_difference:.4f}"
    )

    Path(
        "outputs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    lstm_prediction_df.to_csv(
        LSTM_PREDICTIONS_PATH,
        index=False,
    )

    lstm_engine_metrics_df.to_csv(
        LSTM_ENGINE_METRICS_PATH,
        index=False,
    )

    lstm_rul_slices_df.to_csv(
        LSTM_RUL_SLICES_PATH,
        index=False,
    )

    rf_shared_prediction_df.to_csv(
        RF_SHARED_PREDICTIONS_PATH,
        index=False,
    )

    rf_shared_engine_metrics_df.to_csv(
        RF_SHARED_ENGINE_METRICS_PATH,
        index=False,
    )

    rf_shared_rul_slices_df.to_csv(
        RF_SHARED_RUL_SLICES_PATH,
        index=False,
    )

    comparison_df.to_csv(
        SHARED_COMPARISON_PATH,
        index=False,
    )

    print(
        "\n======================================"
    )
    print(
        "OUTPUTS SAVED"
    )
    print(
        "======================================"
    )

    print(
        LSTM_PREDICTIONS_PATH
    )

    print(
        LSTM_ENGINE_METRICS_PATH
    )

    print(
        LSTM_RUL_SLICES_PATH
    )

    print(
        RF_SHARED_PREDICTIONS_PATH
    )

    print(
        RF_SHARED_ENGINE_METRICS_PATH
    )

    print(
        RF_SHARED_RUL_SLICES_PATH
    )

    print(
        SHARED_COMPARISON_PATH
    )


if __name__ == "__main__":
    main()