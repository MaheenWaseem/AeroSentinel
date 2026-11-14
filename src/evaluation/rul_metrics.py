from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate overall MAE and RMSE.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def build_prediction_table(
    metadata_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Build a row-level prediction table containing
    engine identity, cycle, true RUL, prediction,
    and prediction error.
    """

    prediction_df = metadata_df[
        [
            "engine_id",
            "cycle",
        ]
    ].copy()

    prediction_df[
        "actual_rul"
    ] = np.asarray(
        y_true
    )

    prediction_df[
        "predicted_rul"
    ] = np.asarray(
        y_pred
    )

    prediction_df[
        "error"
    ] = (
        prediction_df["predicted_rul"]
        - prediction_df["actual_rul"]
    )

    prediction_df[
        "absolute_error"
    ] = (
        prediction_df["error"]
        .abs()
    )

    prediction_df[
        "squared_error"
    ] = (
        prediction_df["error"]
        ** 2
    )

    return prediction_df


def calculate_engine_metrics(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate MAE and RMSE separately for every engine.
    """

    engine_metrics_df = (
        prediction_df
        .groupby(
            "engine_id",
            as_index=False,
        )
        .agg(
            engine_mae=(
                "absolute_error",
                "mean",
            ),
            engine_mse=(
                "squared_error",
                "mean",
            ),
        )
    )

    engine_metrics_df[
        "engine_rmse"
    ] = (
        engine_metrics_df[
            "engine_mse"
        ]
        ** 0.5
    )

    engine_metrics_df = (
        engine_metrics_df
        .drop(
            columns=[
                "engine_mse",
            ]
        )
    )

    return engine_metrics_df


def summarize_engine_metrics(
    engine_metrics_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Summarize performance across engines.
    """

    mean_engine_mae = (
        engine_metrics_df[
            "engine_mae"
        ]
        .mean()
    )

    median_engine_mae = (
        engine_metrics_df[
            "engine_mae"
        ]
        .median()
    )

    mean_engine_rmse = (
        engine_metrics_df[
            "engine_rmse"
        ]
        .mean()
    )

    return {
        "mean_engine_mae": float(
            mean_engine_mae
        ),
        "median_engine_mae": float(
            median_engine_mae
        ),
        "mean_engine_rmse": float(
            mean_engine_rmse
        ),
    }