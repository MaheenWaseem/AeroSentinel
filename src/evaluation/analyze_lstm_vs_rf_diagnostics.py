from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LSTM_PREDICTIONS_PATH = Path(
    "outputs/lstm_validation_predictions.csv"
)

RF_PREDICTIONS_PATH = Path(
    "outputs/rf_shared_validation_predictions.csv"
)


SHARED_DIAGNOSTICS_PATH = Path(
    "outputs/lstm_vs_rf_shared_row_diagnostics.csv"
)

ENGINE_DIAGNOSTICS_PATH = Path(
    "outputs/lstm_vs_rf_engine_diagnostics.csv"
)

ENGINE_SUMMARY_PATH = Path(
    "outputs/lstm_vs_rf_engine_summary.csv"
)

HIGH_RUL_ENGINE_DIAGNOSTICS_PATH = Path(
    "outputs/lstm_vs_rf_high_rul_engine_diagnostics.csv"
)

HIGH_RUL_SUBBANDS_PATH = Path(
    "outputs/lstm_vs_rf_high_rul_subbands.csv"
)

HIGH_RUL_CYCLE_BANDS_PATH = Path(
    "outputs/lstm_vs_rf_high_rul_cycle_bands.csv"
)

ENGINE_MAE_DIFFERENCE_PLOT_PATH = Path(
    "outputs/lstm_vs_rf_engine_mae_difference.png"
)

HIGH_RUL_SUBBANDS_PLOT_PATH = Path(
    "outputs/lstm_vs_rf_high_rul_subbands.png"
)


REQUIRED_PREDICTION_COLUMNS = [
    "engine_id",
    "cycle",
    "actual_rul",
    "predicted_rul",
    "error",
    "absolute_error",
    "squared_error",
]


def load_prediction_table(
    path: Path,
    model_name: str,
) -> pd.DataFrame:

    prediction_df = pd.read_csv(
        path
    )

    missing_columns = [
        column
        for column in REQUIRED_PREDICTION_COLUMNS
        if column not in prediction_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{model_name} prediction file is missing "
            f"required columns: {missing_columns}"
        )

    duplicate_mask = prediction_df.duplicated(
        subset=[
            "engine_id",
            "cycle",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = prediction_df.loc[
            duplicate_mask,
            [
                "engine_id",
                "cycle",
            ],
        ]

        raise ValueError(
            f"{model_name} prediction file contains "
            f"duplicate (engine_id, cycle) rows:\n"
            f"{duplicate_rows.head(10)}"
        )

    return prediction_df


def build_shared_prediction_table(
    lstm_predictions_df: pd.DataFrame,
    rf_predictions_df: pd.DataFrame,
) -> pd.DataFrame:

    lstm_df = lstm_predictions_df[
        [
            "engine_id",
            "cycle",
            "actual_rul",
            "predicted_rul",
        ]
    ].copy()

    rf_df = rf_predictions_df[
        [
            "engine_id",
            "cycle",
            "actual_rul",
            "predicted_rul",
        ]
    ].copy()

    lstm_df = lstm_df.rename(
        columns={
            "actual_rul": "lstm_actual_rul",
            "predicted_rul": "lstm_predicted_rul",
        }
    )

    rf_df = rf_df.rename(
        columns={
            "actual_rul": "rf_actual_rul",
            "predicted_rul": "rf_predicted_rul",
        }
    )

    shared_df = lstm_df.merge(
        rf_df,
        on=[
            "engine_id",
            "cycle",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(shared_df) != len(lstm_predictions_df):
        raise ValueError(
            "Not every LSTM prediction row matched an RF prediction row."
        )

    if len(shared_df) != len(rf_predictions_df):
        raise ValueError(
            "Not every RF prediction row matched an LSTM prediction row."
        )

    targets_match = np.allclose(
        shared_df["lstm_actual_rul"].to_numpy(),
        shared_df["rf_actual_rul"].to_numpy(),
    )

    if not targets_match:
        raise ValueError(
            "LSTM and RF actual RUL targets do not match on the shared prediction rows."
        )

    shared_df["actual_rul"] = (
        shared_df["lstm_actual_rul"]
    )

    shared_df = shared_df.drop(
        columns=[
            "lstm_actual_rul",
            "rf_actual_rul",
        ]
    )

    shared_df["lstm_error"] = (
        shared_df["lstm_predicted_rul"]
        - shared_df["actual_rul"]
    )

    shared_df["rf_error"] = (
        shared_df["rf_predicted_rul"]
        - shared_df["actual_rul"]
    )

    shared_df["lstm_absolute_error"] = (
        shared_df["lstm_error"].abs()
    )

    shared_df["rf_absolute_error"] = (
        shared_df["rf_error"].abs()
    )

    shared_df["lstm_squared_error"] = (
        shared_df["lstm_error"] ** 2
    )

    shared_df["rf_squared_error"] = (
        shared_df["rf_error"] ** 2
    )

    shared_df["absolute_error_advantage"] = (
        shared_df["rf_absolute_error"]
        - shared_df["lstm_absolute_error"]
    )

    shared_df = shared_df.sort_values(
        [
            "engine_id",
            "cycle",
        ]
    ).reset_index(
        drop=True
    )

    return shared_df


def calculate_engine_diagnostics(
    shared_df: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, float | int]] = []

    for engine_id, engine_df in shared_df.groupby(
        "engine_id",
        sort=True,
    ):

        lstm_mae = (
            engine_df[
                "lstm_absolute_error"
            ].mean()
        )

        rf_mae = (
            engine_df[
                "rf_absolute_error"
            ].mean()
        )

        lstm_rmse = np.sqrt(
            engine_df[
                "lstm_squared_error"
            ].mean()
        )

        rf_rmse = np.sqrt(
            engine_df[
                "rf_squared_error"
            ].mean()
        )

        lstm_mean_signed_error = (
            engine_df[
                "lstm_error"
            ].mean()
        )

        rf_mean_signed_error = (
            engine_df[
                "rf_error"
            ].mean()
        )

        rows.append(
            {
                "engine_id": int(engine_id),
                "row_count": int(
                    len(engine_df)
                ),
                "lstm_mae": float(
                    lstm_mae
                ),
                "rf_mae": float(
                    rf_mae
                ),
                "rf_mae_minus_lstm_mae": float(
                    rf_mae - lstm_mae
                ),
                "lstm_rmse": float(
                    lstm_rmse
                ),
                "rf_rmse": float(
                    rf_rmse
                ),
                "lstm_mean_signed_error": float(
                    lstm_mean_signed_error
                ),
                "rf_mean_signed_error": float(
                    rf_mean_signed_error
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_engine_comparison(
    engine_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:

    difference = (
        engine_diagnostics_df[
            "rf_mae_minus_lstm_mae"
        ]
    )

    tolerance = 1e-12

    lstm_wins = int(
        (
            difference
            > tolerance
        ).sum()
    )

    rf_wins = int(
        (
            difference
            < -tolerance
        ).sum()
    )

    ties = int(
        (
            difference.abs()
            <= tolerance
        ).sum()
    )

    summary = {
        "engine_count": int(
            len(engine_diagnostics_df)
        ),
        "lstm_lower_mae_engine_count": (
            lstm_wins
        ),
        "rf_lower_mae_engine_count": (
            rf_wins
        ),
        "tie_engine_count": ties,
        "mean_rf_mae_minus_lstm_mae": float(
            difference.mean()
        ),
        "median_rf_mae_minus_lstm_mae": float(
            difference.median()
        ),
        "lstm_mean_engine_mae": float(
            engine_diagnostics_df[
                "lstm_mae"
            ].mean()
        ),
        "rf_mean_engine_mae": float(
            engine_diagnostics_df[
                "rf_mae"
            ].mean()
        ),
        "lstm_median_engine_mae": float(
            engine_diagnostics_df[
                "lstm_mae"
            ].median()
        ),
        "rf_median_engine_mae": float(
            engine_diagnostics_df[
                "rf_mae"
            ].median()
        ),
    }

    return pd.DataFrame(
        [summary]
    )


def calculate_high_rul_engine_diagnostics(
    shared_df: pd.DataFrame,
) -> pd.DataFrame:

    high_rul_df = shared_df.loc[
        shared_df["actual_rul"] > 100
    ].copy()

    rows: list[dict[str, float | int]] = []

    for engine_id, engine_df in high_rul_df.groupby(
        "engine_id",
        sort=True,
    ):

        lstm_mae = (
            engine_df[
                "lstm_absolute_error"
            ].mean()
        )

        rf_mae = (
            engine_df[
                "rf_absolute_error"
            ].mean()
        )

        lstm_mean_signed_error = (
            engine_df[
                "lstm_error"
            ].mean()
        )

        rf_mean_signed_error = (
            engine_df[
                "rf_error"
            ].mean()
        )

        rows.append(
            {
                "engine_id": int(engine_id),
                "high_rul_row_count": int(
                    len(engine_df)
                ),
                "lstm_high_rul_mae": float(
                    lstm_mae
                ),
                "rf_high_rul_mae": float(
                    rf_mae
                ),
                "rf_mae_minus_lstm_mae": float(
                    rf_mae - lstm_mae
                ),
                "lstm_high_rul_mean_signed_error": float(
                    lstm_mean_signed_error
                ),
                "rf_high_rul_mean_signed_error": float(
                    rf_mean_signed_error
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def add_high_rul_subband(
    shared_df: pd.DataFrame,
) -> pd.DataFrame:

    high_rul_df = shared_df.loc[
        shared_df["actual_rul"] > 100
    ].copy()

    high_rul_df["high_rul_subband"] = pd.cut(
        high_rul_df["actual_rul"],
        bins=[
            100,
            150,
            200,
            np.inf,
        ],
        labels=[
            "rul_101_150",
            "rul_151_200",
            "rul_above_200",
        ],
        right=True,
        include_lowest=False,
    )

    return high_rul_df


def summarize_grouped_diagnostics(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    rows: list[dict[str, float | int | str]] = []

    for group_value, group_df in dataframe.groupby(
        group_column,
        observed=True,
        sort=False,
    ):

        rows.append(
            {
                group_column: str(group_value),
                "row_count": int(
                    len(group_df)
                ),
                "engine_count": int(
                    group_df[
                        "engine_id"
                    ].nunique()
                ),
                "mean_actual_rul": float(
                    group_df[
                        "actual_rul"
                    ].mean()
                ),
                "lstm_mae": float(
                    group_df[
                        "lstm_absolute_error"
                    ].mean()
                ),
                "rf_mae": float(
                    group_df[
                        "rf_absolute_error"
                    ].mean()
                ),
                "rf_mae_minus_lstm_mae": float(
                    group_df[
                        "rf_absolute_error"
                    ].mean()
                    - group_df[
                        "lstm_absolute_error"
                    ].mean()
                ),
                "lstm_mean_signed_error": float(
                    group_df[
                        "lstm_error"
                    ].mean()
                ),
                "rf_mean_signed_error": float(
                    group_df[
                        "rf_error"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def add_high_rul_cycle_band(
    high_rul_df: pd.DataFrame,
) -> pd.DataFrame:

    cycle_df = high_rul_df.copy()

    cycle_df["cycle_band"] = pd.cut(
        cycle_df["cycle"],
        bins=[
            29,
            49,
            79,
            np.inf,
        ],
        labels=[
            "cycle_30_49",
            "cycle_50_79",
            "cycle_80_plus",
        ],
        right=True,
    )

    return cycle_df


def calculate_regression_compression(
    high_rul_df: pd.DataFrame,
) -> dict[str, float]:

    actual_rul = high_rul_df[
        "actual_rul"
    ].to_numpy()

    lstm_predictions = high_rul_df[
        "lstm_predicted_rul"
    ].to_numpy()

    rf_predictions = high_rul_df[
        "rf_predicted_rul"
    ].to_numpy()

    lstm_slope, lstm_intercept = np.polyfit(
        actual_rul,
        lstm_predictions,
        1,
    )

    rf_slope, rf_intercept = np.polyfit(
        actual_rul,
        rf_predictions,
        1,
    )

    return {
        "lstm_prediction_vs_actual_slope": float(
            lstm_slope
        ),
        "lstm_prediction_vs_actual_intercept": float(
            lstm_intercept
        ),
        "rf_prediction_vs_actual_slope": float(
            rf_slope
        ),
        "rf_prediction_vs_actual_intercept": float(
            rf_intercept
        ),
    }


def save_engine_mae_difference_plot(
    engine_diagnostics_df: pd.DataFrame,
) -> None:

    plot_df = engine_diagnostics_df.sort_values(
        "rf_mae_minus_lstm_mae"
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df[
            "engine_id"
        ].astype(str),
        plot_df[
            "rf_mae_minus_lstm_mae"
        ],
    )

    plt.axvline(
        0.0,
        linewidth=1,
    )

    plt.xlabel(
        "RF MAE - LSTM MAE"
    )

    plt.ylabel(
        "Engine ID"
    )

    plt.title(
        "FD001 Per-Engine LSTM Advantage"
    )

    plt.tight_layout()

    plt.savefig(
        ENGINE_MAE_DIFFERENCE_PLOT_PATH,
        dpi=150,
    )

    plt.close()


def save_high_rul_subband_plot(
    high_rul_subbands_df: pd.DataFrame,
) -> None:

    x_positions = np.arange(
        len(high_rul_subbands_df)
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x_positions - width / 2,
        high_rul_subbands_df[
            "lstm_mean_signed_error"
        ],
        width,
        label="LSTM",
    )

    plt.bar(
        x_positions + width / 2,
        high_rul_subbands_df[
            "rf_mean_signed_error"
        ],
        width,
        label="Random Forest",
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xticks(
        x_positions,
        high_rul_subbands_df[
            "high_rul_subband"
        ],
    )

    plt.xlabel(
        "High-RUL Band"
    )

    plt.ylabel(
        "Mean Signed Error"
    )

    plt.title(
        "FD001 High-RUL Signed Error by Lifecycle Band"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        HIGH_RUL_SUBBANDS_PLOT_PATH,
        dpi=150,
    )

    plt.close()


def print_top_engine_results(
    engine_diagnostics_df: pd.DataFrame,
) -> None:

    print(
        "\n======================================"
    )
    print(
        "BIGGEST LSTM GAINS"
    )
    print(
        "======================================"
    )

    biggest_gains = (
        engine_diagnostics_df
        .sort_values(
            "rf_mae_minus_lstm_mae",
            ascending=False,
        )
        .head(5)
    )

    print(
        biggest_gains[
            [
                "engine_id",
                "row_count",
                "lstm_mae",
                "rf_mae",
                "rf_mae_minus_lstm_mae",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "BIGGEST LSTM REGRESSIONS"
    )
    print(
        "======================================"
    )

    biggest_regressions = (
        engine_diagnostics_df
        .sort_values(
            "rf_mae_minus_lstm_mae",
            ascending=True,
        )
        .head(5)
    )

    print(
        biggest_regressions[
            [
                "engine_id",
                "row_count",
                "lstm_mae",
                "rf_mae",
                "rf_mae_minus_lstm_mae",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "WORST LSTM ENGINES"
    )
    print(
        "======================================"
    )

    worst_lstm = (
        engine_diagnostics_df
        .sort_values(
            "lstm_mae",
            ascending=False,
        )
        .head(5)
    )

    print(
        worst_lstm[
            [
                "engine_id",
                "lstm_mae",
                "lstm_mean_signed_error",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "WORST RF ENGINES"
    )
    print(
        "======================================"
    )

    worst_rf = (
        engine_diagnostics_df
        .sort_values(
            "rf_mae",
            ascending=False,
        )
        .head(5)
    )

    print(
        worst_rf[
            [
                "engine_id",
                "rf_mae",
                "rf_mean_signed_error",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:

    print(
        "\n======================================"
    )
    print(
        "FD001 LSTM VS RF DIAGNOSTICS"
    )
    print(
        "======================================"
    )

    lstm_predictions_df = load_prediction_table(
        LSTM_PREDICTIONS_PATH,
        "LSTM",
    )

    rf_predictions_df = load_prediction_table(
        RF_PREDICTIONS_PATH,
        "Random Forest",
    )

    shared_df = build_shared_prediction_table(
        lstm_predictions_df,
        rf_predictions_df,
    )

    print(
        "\nShared rows:"
    )
    print(
        len(shared_df)
    )

    print(
        "\nShared engines:"
    )
    print(
        shared_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nShared target check:"
    )
    print(
        "PASSED"
    )

    engine_diagnostics_df = (
        calculate_engine_diagnostics(
            shared_df
        )
    )

    engine_summary_df = (
        summarize_engine_comparison(
            engine_diagnostics_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "PER-ENGINE SUMMARY"
    )
    print(
        "======================================"
    )

    print(
        engine_summary_df.to_string(
            index=False
        )
    )

    print_top_engine_results(
        engine_diagnostics_df
    )

    high_rul_engine_df = (
        calculate_high_rul_engine_diagnostics(
            shared_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "HIGH-RUL ENGINE SUMMARY"
    )
    print(
        "======================================"
    )

    print(
        "\nHigh-RUL engines:"
    )
    print(
        len(high_rul_engine_df)
    )

    print(
        "\nEngines with LSTM high-RUL underprediction:"
    )
    print(
        int(
            (
                high_rul_engine_df[
                    "lstm_high_rul_mean_signed_error"
                ]
                < 0
            ).sum()
        )
    )

    print(
        "\nEngines with RF high-RUL underprediction:"
    )
    print(
        int(
            (
                high_rul_engine_df[
                    "rf_high_rul_mean_signed_error"
                ]
                < 0
            ).sum()
        )
    )

    print(
        "\nEngines where LSTM has lower high-RUL MAE:"
    )
    print(
        int(
            (
                high_rul_engine_df[
                    "rf_mae_minus_lstm_mae"
                ]
                > 0
            ).sum()
        )
    )

    print(
        "\nEngines where RF has lower high-RUL MAE:"
    )
    print(
        int(
            (
                high_rul_engine_df[
                    "rf_mae_minus_lstm_mae"
                ]
                < 0
            ).sum()
        )
    )

    print(
        "\nMost negative LSTM high-RUL signed errors:"
    )

    print(
        high_rul_engine_df
        .sort_values(
            "lstm_high_rul_mean_signed_error"
        )
        .head(5)[
            [
                "engine_id",
                "high_rul_row_count",
                "lstm_high_rul_mae",
                "rf_high_rul_mae",
                "lstm_high_rul_mean_signed_error",
                "rf_high_rul_mean_signed_error",
            ]
        ]
        .to_string(
            index=False
        )
    )

    high_rul_df = add_high_rul_subband(
        shared_df
    )

    high_rul_subbands_df = (
        summarize_grouped_diagnostics(
            high_rul_df,
            "high_rul_subband",
        )
    )

    print(
        "\n======================================"
    )
    print(
        "HIGH-RUL SUB-BANDS"
    )
    print(
        "======================================"
    )

    print(
        high_rul_subbands_df.to_string(
            index=False
        )
    )

    high_rul_cycle_df = (
        add_high_rul_cycle_band(
            high_rul_df
        )
    )

    high_rul_cycle_bands_df = (
        summarize_grouped_diagnostics(
            high_rul_cycle_df,
            "cycle_band",
        )
    )

    print(
        "\n======================================"
    )
    print(
        "HIGH-RUL BY ENDING CYCLE"
    )
    print(
        "======================================"
    )

    print(
        high_rul_cycle_bands_df.to_string(
            index=False
        )
    )

    regression_compression = (
        calculate_regression_compression(
            high_rul_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "HIGH-RUL PREDICTION COMPRESSION"
    )
    print(
        "======================================"
    )

    print(
        "\nLSTM prediction-vs-actual slope:"
    )
    print(
        f"{regression_compression['lstm_prediction_vs_actual_slope']:.4f}"
    )

    print(
        "\nRF prediction-vs-actual slope:"
    )
    print(
        f"{regression_compression['rf_prediction_vs_actual_slope']:.4f}"
    )

    print(
        "\nInterpret slopes cautiously."
    )
    print(
        "A slope below 1 can indicate compression "
        "of predictions across the high-RUL range, "
        "but it does not establish a causal mechanism."
    )

    SHARED_DIAGNOSTICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shared_df.to_csv(
        SHARED_DIAGNOSTICS_PATH,
        index=False,
    )

    engine_diagnostics_df.to_csv(
        ENGINE_DIAGNOSTICS_PATH,
        index=False,
    )

    engine_summary_df.to_csv(
        ENGINE_SUMMARY_PATH,
        index=False,
    )

    high_rul_engine_df.to_csv(
        HIGH_RUL_ENGINE_DIAGNOSTICS_PATH,
        index=False,
    )

    high_rul_subbands_df.to_csv(
        HIGH_RUL_SUBBANDS_PATH,
        index=False,
    )

    high_rul_cycle_bands_df.to_csv(
        HIGH_RUL_CYCLE_BANDS_PATH,
        index=False,
    )

    save_engine_mae_difference_plot(
        engine_diagnostics_df
    )

    save_high_rul_subband_plot(
        high_rul_subbands_df
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
        SHARED_DIAGNOSTICS_PATH
    )

    print(
        ENGINE_DIAGNOSTICS_PATH
    )

    print(
        ENGINE_SUMMARY_PATH
    )

    print(
        HIGH_RUL_ENGINE_DIAGNOSTICS_PATH
    )

    print(
        HIGH_RUL_SUBBANDS_PATH
    )

    print(
        HIGH_RUL_CYCLE_BANDS_PATH
    )

    print(
        ENGINE_MAE_DIFFERENCE_PLOT_PATH
    )

    print(
        HIGH_RUL_SUBBANDS_PLOT_PATH
    )


if __name__ == "__main__":
    main()